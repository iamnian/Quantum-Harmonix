import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import Counter
import networkx as nx
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy
import librosa
import sys

try:
    import IPython.display as ipd
except ImportError:
    print("Warning: IPython not installed - audio playback disabled")
    ipd = None
# ======================
# ENHANCED CONSTANTS
# ======================
intervals = {
    0: {"name": "Unison", "tension": 0, "type": "consonant", "weight": 0.8},
    1: {"name": "m2", "tension": 4, "type": "dissonant", "weight": 0.6},
    2: {"name": "M2", "tension": 3, "type": "dissonant", "weight": 0.7},
    3: {"name": "m3", "tension": 1, "type": "consonant", "weight": 0.9},
    4: {"name": "M3", "tension": 1, "type": "consonant", "weight": 0.9},
    5: {"name": "P4", "tension": 2, "type": "consonant", "weight": 0.8},
    6: {"name": "TT", "tension": 5, "type": "dissonant", "weight": 0.5},
    7: {"name": "P5", "tension": 0, "type": "consonant", "weight": 1.0},
    8: {"name": "m6", "tension": 2, "type": "consonant", "weight": 0.7},
    9: {"name": "M6", "tension": 2, "type": "consonant", "weight": 0.7},
    10: {"name": "m7", "tension": 3, "type": "dissonant", "weight": 0.6},
    11: {"name": "M7", "tension": 4, "type": "dissonant", "weight": 0.5},
    12: {"name": "P8", "tension": 0, "type": "consonant", "weight": 0.8},
    # Extended intervals
    13: {"name": "m9", "tension": 4, "type": "dissonant", "weight": 0.4},
    14: {"name": "M9", "tension": 3, "type": "dissonant", "weight": 0.5},
    15: {"name": "m10", "tension": 2, "type": "consonant", "weight": 0.6},
    16: {"name": "M10", "tension": 2, "type": "consonant", "weight": 0.6},
    17: {"name": "P11", "tension": 1, "type": "consonant", "weight": 0.7},
    19: {"name": "P12", "tension": 0, "type": "consonant", "weight": 0.7}
}

note_names = ["E","F","F#","Gb","G","G#","Ab","A","A#","Bb","B","C","C#","Db","D","D#","Eb"]
note_frequencies = {
    'Db': 77.78, 'D': 82.41, 'Eb': 87.31, 'E': 82.41,
    'F': 87.31, 'F#': 92.50, 'G': 98.00, 'G#': 103.83,
    'A': 110.00, 'A#': 116.54, 'B': 123.47, 'C': 130.81,
    'C#': 138.59, 'D': 146.83, 'D#': 155.56
}

SCALE_TYPES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10]
}

DATABASE_FILE = "quantum_tunings_db.csv"

# ======================
# ENHANCED CORE FUNCTIONS
# ======================
def init_database():
    """Initialize or load the tuning database with new features"""
    if os.path.exists(DATABASE_FILE):
        try:
            db = pd.read_csv(DATABASE_FILE)
            # Convert stringified lists back to actual lists
            if 'notes' in db.columns:
                db['notes'] = db['notes'].apply(eval)
            return db
        except Exception as e:
            print(f"Error loading database: {e}, creating new one")
            return pd.DataFrame(columns=[
                'tuning_id', 'date_created', 'notes', 'intervals', 
                'interval_values', 'rating', 'review', 'times_tried',
                'tension_score', 'harmonic_type', 'root_note',
                'modal_quality', 'entropy', 'scale_similarity'
            ])
    else:
        return pd.DataFrame(columns=[
            'tuning_id', 'date_created', 'notes', 'intervals', 
            'interval_values', 'rating', 'review', 'times_tried',
            'tension_score', 'harmonic_type', 'root_note',
            'modal_quality', 'entropy', 'scale_similarity'
        ])
def get_note_at_position(string_note, fret):
    """Get note name at specific fret position"""
    # Standardize note name to sharp notation first
    note_map = {
        'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#',
        'F#': 'F#', 'G#': 'G#', 'A#': 'A#', 'C#': 'C#', 'D#': 'D#'
    }
    
    # Convert input note to sharp notation for calculation
    if string_note in note_map:
        string_note = note_map[string_note]
    
    root_index = note_names.index(string_note)
    result_note = note_names[(root_index + fret) % 12]
    
    # Return in the same notation as the input
    if string_note.endswith('b') and result_note in note_map.values():
        # Find the flat equivalent
        for flat, sharp in note_map.items():
            if sharp == result_note:
                return flat
    return result_note

def is_note_on_string(string_note, target_note, start_fret, span=4):
    """Check if note exists within fret span"""
    for fret in range(start_fret, start_fret + span):
        current_note = get_note_at_position(string_note, fret)
        # Compare notes directly (both will be in consistent notation)
        if current_note == target_note:
            return True
    return False

def get_scale_notes(root_note, scale_name):
    """Get note names for a scale"""
    root_index = note_names.index(root_note)
    return [note_names[(root_index + ivl) % 12] for ivl in SCALE_TYPES[scale_name]]




def tuning_exists(db, tuning_notes):
    """Check if tuning already exists in database"""
    if db.empty:
        return False
    tuning_str = "-".join(tuning_notes)
    existing_tunings = db['notes'].apply(lambda x: "-".join(x))
    return tuning_str in existing_tunings.values

def generate_tuning(max_semitone_change=6, flavor_bias=None, root_note=None):
    # Standardize root note naming
    """Generate tunings with exotic intervals and flavor control"""
    # Flavor weighting
    flavor_weights = {
        "major": [3,4,5,7,9,16],  # Major third, sixth, tenth
        "minor": [3,5,7,8,10,15],  # Minor third, sixth, seventh, tenth
        "exotic": [6,10,11,13,14]  # Tritones, extended intervals
    }
    
    root_note = root_note if root_note else np.random.choice(['Db','D','Eb','E','F','F#','G'])
    root_freq = note_frequencies[root_note]
    
    # Dynamic interval selection based on flavor
    if flavor_bias:
        candidates = flavor_weights[flavor_bias]
        weights = [intervals[ivl]["weight"] if ivl in intervals else 0.5 for ivl in candidates]
    else:
        candidates = [k for k in intervals.keys() if 1 <= k <= max_semitone_change]
        weights = [intervals[k]["weight"] for k in candidates]
    
    weights = np.array(weights) / np.sum(weights)
    random_intervals = np.random.choice(candidates, size=5, p=weights, replace=False)
    
    notes = [root_note]
    current_freq = root_freq
    
    for interval in random_intervals:
        current_freq *= (2 ** (interval / 12))
        note_num = round(np.log2(current_freq / 440) * 12) % 12
        notes.append(note_names[note_num % 12])
    
    interval_names = [intervals[num]["name"] for num in random_intervals]
    
    # Calculate additional properties
    tension = sum([intervals[num]["tension"] for num in random_intervals])
    harmonic_type = analyze_harmony(random_intervals)
    modal_quality = calculate_modal_quality(random_intervals)
    tuning_entropy = calculate_entropy(notes)
    scale_sim = calculate_scale_similarity(notes)
    
    return {
        'notes': notes,
        'interval_nums': random_intervals,
        'interval_names': interval_names,
        'root_note': root_note,
        'tension_score': tension,
        'harmonic_type': harmonic_type,
        'modal_quality': modal_quality,
        'entropy': tuning_entropy,
        'scale_similarity': scale_sim
    }

def analyze_harmony(intervals_list):
    """Classify tuning's harmonic properties"""
    interval_types = [intervals[ivl]["type"] for ivl in intervals_list]
    consonant = interval_types.count("consonant")
    dissonant = interval_types.count("dissonant")
    
    if consonant > dissonant + 2:
        return "Consonant"
    elif dissonant > consonant + 2:
        return "Dissonant"
    else:
        return "Mixed"

def calculate_modal_quality(intervals_list):
    """Determine if tuning favors major/minor/other scales"""
    major_count = sum(1 for i in intervals_list if intervals.get(i,{}).get('name','')[0]=='M')
    minor_count = sum(1 for i in intervals_list if intervals.get(i,{}).get('name','')[0]=='m')
    
    if major_count > minor_count + 1:
        return "Major倾向"
    elif minor_count > major_count + 1:
        return "Minor倾向"
    else:
        return "Neutral"

def calculate_entropy(notes):
    """Measure of tuning's unpredictability"""
    unique_notes = len(set(notes))
    return unique_notes / len(notes)

def calculate_scale_similarity(notes):
    """Calculate similarity to known scales"""
    note_set = set(notes)
    max_similarity = 0
    
    for scale_name, scale_notes in SCALE_TYPES.items():
        scale_note_set = set([note_names[i % 12] for i in scale_notes])
        similarity = len(note_set & scale_note_set) / len(note_set | scale_note_set)
        if similarity > max_similarity:
            max_similarity = similarity
    
    return max_similarity

def audio_analysis(tuning):
    """Perform basic audio analysis on the tuning"""
    freqs = [note_frequencies[note] for note in tuning['notes']]
    
    # Calculate harmonicity score
    ratios = []
    for i in range(1, len(freqs)):
        ratios.append(freqs[i]/freqs[0])
    
    harmonicity = 1 / np.std(ratios)
    
    # Calculate roughness (simplified)
    roughness = 0
    for i in range(len(freqs)):
        for j in range(i+1, len(freqs)):
            f1, f2 = sorted([freqs[i], freqs[j]])
            roughness += 0.5 * (f2 - f1) / (f1 * 0.25)
    
    return {
        'harmonicity': harmonicity,
        'roughness': roughness,
        'freq_variation': np.std(freqs)
    }

# ======================
# ENHANCED VISUALIZATION
# ======================
def visualize_tunings(db):
    """Expanded visualization suite"""
    if db.empty:
        print("No tunings to visualize!")
        return
    
    try:
        plt.figure(figsize=(20, 16))
        
        # 1. Interval Distribution
        plt.subplot(3, 3, 1)
        all_intervals = []
        for ivl_str in db['intervals']:
            all_intervals.extend(ivl_str.split('-'))
        interval_counts = pd.Series(all_intervals).value_counts()
        sns.barplot(x=interval_counts.index, y=interval_counts.values, palette="viridis")
        plt.title("Interval Frequency")
        plt.xticks(rotation=45)
        
        # 2. Root Note Distribution
        plt.subplot(3, 3, 2)
        root_counts = db['root_note'].value_counts()
        sns.barplot(x=root_counts.index, y=root_counts.values, palette="magma")
        plt.title("Root Note Distribution")
        
        # 3. Tension Analysis
        plt.subplot(3, 3, 3)
        sns.histplot(db['tension_score'], bins=10, kde=True)
        plt.title("Tension Distribution")
        
        # 4. Harmonic Type
        plt.subplot(3, 3, 4)
        type_counts = db['harmonic_type'].value_counts()
        sns.barplot(x=type_counts.index, y=type_counts.values, palette="plasma")
        plt.title("Harmonic Character")
        
        # 5. Network Graph
        plt.subplot(3, 3, 5)
        G = nx.Graph()
        for tuning in db['notes']:
            for i in range(len(tuning)-1):
                G.add_edge(tuning[i], tuning[i+1])
        nx.draw(G, with_labels=True, node_size=800, node_color='skyblue')
        plt.title("Note Transition Network")
        
        # 6. Cluster Analysis
        plt.subplot(3, 3, 6)
        if len(db) >= 3:
            features = db['interval_values'].apply(
                lambda x: [int(num) for num in x.split('-')]).tolist()
            kmeans = KMeans(n_clusters=min(3, len(db)))
            db['cluster'] = kmeans.fit_predict(features)
            sns.scatterplot(data=db, x='tension_score', y='rating', hue='cluster', palette="deep")
            plt.title("Tuning Clusters")
        
        # 7. Modal Quality
        plt.subplot(3, 3, 7)
        quality_counts = db['modal_quality'].value_counts()
        quality_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title("Modal Quality Distribution")
        
        # 8. Entropy Analysis
        plt.subplot(3, 3, 8)
        sns.boxplot(x='modal_quality', y='entropy', data=db)
        plt.title("Note Diversity by Modal Quality")
        
        # 9. Scale Similarity
        plt.subplot(3, 3, 9)
        sns.histplot(db['scale_similarity'], bins=10, kde=True)
        plt.title("Scale Similarity Scores")
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")

# ======================
# ENHANCED MAIN FUNCTION
# ======================
def main():
    db = init_database()
    
    print("🎸 QUANTUM TUNING GENERATOR 3.0 🎸")
    print("---------------------------------")
    print("NEW FEATURES:")
    print("- Auto-fretboard visualization")
    print("- Chord shape detection")
    print("- Advanced tuning saving")
    print("- Interactive exploration\n")
    
    while True:
        print("\nOptions:")
        print("1. Generate new tunings")
        print("2. View existing tunings")
        print("3. Rate a tuning")
        print("4. Visualize data")
        print("5. Export tunings")
        print("6. Advanced generation")
        print("7. Exit")
        
        choice = input("Enter your choice (1-7): ")
        
        if choice == "1":
            num_tunings = int(input("How many tunings to generate? (1-20): "))
            num_tunings = max(1, min(20, num_tunings))
            
            for _ in range(num_tunings):
                tuning = generate_tuning()
                visualize_tuning_with_chords(tuning)  # Auto-show fretboard
                
                if not tuning_exists(db, tuning['notes']):
                    print_tuning_details(tuning)
                    
                    save = input("Save this tuning? (y/n): ").lower()
                    if save == 'y':
                        db = save_tuning(db, tuning)
                        print("Tuning saved with chord shapes!")
            
            db.to_csv(DATABASE_FILE, index=False)
        
        elif choice == "2":
            if db.empty:
                print("No tunings in database yet!")
                continue
                
            print("\nExisting Tunings:")
            for idx, row in db.iterrows():
                print(f"\nTuning ID: {row['tuning_id']}")
                print(f"Created: {row['date_created']}")
                print(f"Notes: {'-'.join(row['notes'])}")
                print(f"Intervals: {row['intervals']}")
                print(f"Rating: {row['rating'] or 'Not rated'}")
                print("-"*40)
        
        elif choice == "3":
            if db.empty:
                print("No tunings to rate!")
                continue
                
            tuning_id = int(input("Enter tuning ID to rate: "))
            if tuning_id not in db['tuning_id'].values:
                print("Invalid tuning ID!")
                continue
                
            rating = float(input("Enter rating (1-5): "))
            review = input("Enter review (optional): ")
            
            db.loc[db['tuning_id'] == tuning_id, 'rating'] = rating
            db.loc[db['tuning_id'] == tuning_id, 'review'] = review
            db.to_csv(DATABASE_FILE, index=False)
            print("Rating saved!")
        
        elif choice == "4":
            visualize_tunings(db)
        
        elif choice == "5":
            if db.empty:
                print("No tunings to export!")
                continue
                
            filename = input("Enter filename to export to (e.g., 'my_tunings.csv'): ")
            db.to_csv(filename, index=False)
            print(f"Tunings exported to {filename}")
        
        elif choice == "6":
            print("\nAdvanced Generation Options:")
            print("1. Specify root note")
            print("2. Set flavor bias")
            print("3. Generate with audio analysis")
            adv_choice = input("Enter choice (1-3): ")
            
            tuning = None
            if adv_choice == "1":
                root = input("Enter root note (e.g., 'Eb', 'F#'): ").capitalize()
                if root not in note_frequencies:
                    print("Invalid note!")
                    continue
                tuning = generate_tuning(root_note=root)
                
            elif adv_choice == "2":
                flavor = input("Enter flavor (major/minor/exotic): ").lower()
                if flavor not in ['major', 'minor', 'exotic']:
                    print("Invalid flavor!")
                    continue
                tuning = generate_tuning(flavor_bias=flavor)
                
            elif adv_choice == "3":
                tuning = generate_tuning()
                audio = audio_analysis(tuning)
                print("\nAudio Analysis:")
                print(f"Harmonicity: {audio['harmonicity']:.2f}")
                print(f"Roughness: {audio['roughness']:.2f}")
            
            if tuning:
                visualize_tuning_with_chords(tuning)  # Auto-show fretboard
                if not tuning_exists(db, tuning['notes']):
                    save = input("Save this advanced tuning? (y/n): ").lower()
                    if save == 'y':
                        db = save_tuning(db, tuning)
                        print("Advanced tuning saved!")
        
        elif choice == "7":
            print("Exiting Quantum Tuner...")
            break
        
        else:
            print("Invalid choice! Please enter 1-7")

# [Previous helper functions remain the same until end]

def visualize_tuning_with_chords(tuning):
    """Automatically display fretboard with discovered chords"""
    chords = find_chord_shapes(tuning)
    print("\nDiscovered Chord Shapes:")
    for name, positions in chords.items():
        print(f"{name}: {positions[:2]}")  # Show first 2 voicings
    
    plt.figure(figsize=(18, 8))
    plot_fretboard(tuning, chords=chords)
    plt.show()
    
    # Additional visualization
    plot_tension_heatmap(tuning)
    plt.show()

def save_tuning(db, tuning):
    """Save tuning with all metadata to database"""
    new_entry = {
        'tuning_id': len(db) + 1,
        'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'notes': tuning['notes'],
        'intervals': "-".join(tuning['interval_names']),
        'interval_values': "-".join(map(str, tuning['interval_nums'])),
        'rating': None,
        'review': None,
        'times_tried': 0,
        'root_note': tuning['root_note'],
        'tension_score': tuning['tension_score'],
        'harmonic_type': tuning['harmonic_type'],
        'modal_quality': tuning['modal_quality'],
        'entropy': tuning['entropy'],
        'scale_similarity': tuning['scale_similarity']
    }
    return pd.concat([db, pd.DataFrame([new_entry])], ignore_index=True)

def demo_mode():
    """Auto-run presentation demo"""
    print("\n=== DEMO MODE ===")
    tuning = generate_tuning(root_note="D", flavor_bias="exotic")
    
    print("\n1. Generated Tuning:")
    print_tuning_details(tuning)
    
    print("\n2. Audio Analysis:")
    audio = audio_analysis(tuning)
    print(f"Harmonicity: {audio['harmonicity']:.2f}")
    
    print("\n3. Visualization:")
    visualize_tuning_with_chords(tuning)

def print_tuning_details(tuning):
    """Helper function to print tuning details"""
    print(f"\nGenerated Tuning:")
    print(f"Root: {tuning['root_note']}")
    print(f"Notes: {' - '.join(tuning['notes'])}")
    print(f"Intervals: {' -> '.join(tuning['interval_names'])}")
    print(f"Tension: {tuning['tension_score']}")
    print(f"Harmony: {tuning['harmonic_type']}")
    print(f"Modal Quality: {tuning['modal_quality']}")
    print(f"Scale Similarity: {tuning['scale_similarity']:.2f}")

def plot_fretboard(tuning, chords=None, scale=None):
    """Generate interactive fretboard diagrams for any tuning"""
    strings = tuning['notes']
    frets = 24
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Create fretboard grid
    for fret in range(frets+1):
        ax.axvline(fret, color='black', lw=1 if fret%12 else 3)
    for i, string in enumerate(strings):
        ax.hlines(i, 0, frets, color='gray', lw=6)
        ax.text(-1, i, string, ha='right', va='center', fontsize=14, weight='bold')
    
    # Mark chord shapes if provided
    if chords:
        for chord_name, positions in chords.items():
            for string, fret in positions:
                circle = plt.Circle((fret, string), 0.3, color='red')
                ax.add_patch(circle)
                ax.text(fret, string, chord_name[0], color='white', 
                       ha='center', va='center', fontsize=10)
    
    # Highlight scale patterns
    if scale:
        scale_notes = get_scale_notes(tuning['root_note'], scale)
        for fret in range(frets):
            for string in range(len(strings)):
                note = get_note_at_position(strings[string], fret)
                if note in scale_notes:
                    ax.add_patch(plt.Circle((fret+0.5, string), 0.2, 
                                color='blue', alpha=0.3))
    
    ax.set_xlim(-2, frets+1)
    ax.set_ylim(-1, len(strings))
    ax.axis('off')
    plt.title(f"Fretboard for {'-'.join(strings)} Tuning", fontsize=16)
    plt.tight_layout()
    return fig


def find_chord_shapes(tuning):
    """Automatically discover playable chord shapes for any tuning"""
    base_chords = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        '7th': [0, 4, 7, 10],
        'sus4': [0, 5, 7]
    }
    
    found_shapes = {}
    for name, intervals in base_chords.items():
        shapes = []
        # Algorithm to find playable voicings
        for start_fret in range(12):
            positions = []
            for string in range(6):
                target_note = (intervals[string%len(intervals)] + start_fret) % 12
                # Check if note exists within 4-fret span
                if is_note_on_string(tuning['notes'][string], target_note, start_fret, 4):
                    positions.append((string, start_fret + (target_note//12)))
            if len(positions) >= 3:  # Minimum 3-note chord
                shapes.append(positions)
        found_shapes[name] = shapes[:3]  # Return top 3 voicings
    
    return found_shapes

def generate_scale_patterns(tuning, scale_name='major'):
    """Show all positions of a scale on the fretboard"""
    scale_intervals = SCALE_TYPES[scale_name]
    root_index = note_names.index(tuning['root_note'])
    scale_notes = [note_names[(root_index + ivl)%12] for ivl in scale_intervals]
    
    # Generate CAGED-style patterns
    patterns = []
    for position in range(5):
        pattern = []
        for string in range(6):
            string_notes = []
            for fret in range(12):
                note = get_note_at_position(tuning['notes'][string], fret)
                if note in scale_notes:
                    string_notes.append(fret)
            pattern.append(string_notes)
        patterns.append(pattern)
    
    return patterns

def plot_tension_heatmap(tuning):
    """Show tension distribution across the fretboard"""
    tension_map = np.zeros((6, 24))
    
    for string in range(6):
        root_note = tuning['notes'][string]
        for fret in range(24):
            interval = (note_names.index(get_note_at_position(root_note, fret)) - 
                       note_names.index(root_note)) % 12
            tension_map[string, fret] = intervals.get(interval, {}).get('tension', 0)
    
    plt.figure(figsize=(20, 6))
    sns.heatmap(tension_map, annot=True, fmt=".1f", cmap="YlOrRd",
                yticklabels=tuning['notes'])
    plt.title("Fretboard Tension Heatmap")
    plt.xlabel("Fret")
    plt.ylabel("String")
    
def synthesize_tuning(tuning):
    """Generate playable audio samples"""
    freqs = [note_frequencies[note] for note in tuning['notes']]
    duration = 2.0
    sample_rate = 44100
    
    audio = np.zeros(int(duration * sample_rate))
    for freq in freqs:
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        audio += 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-0.5 * t)
    
    return ipd.Audio(audio, rate=sample_rate)


if __name__ == "__main__":
    # Add this at the start of main()
    if "--demo" in sys.argv:
        demo_mode()
        sys.exit()
    main()