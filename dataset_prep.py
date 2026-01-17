"""
UrbanSound8K Dataset Preparation for Edge Impulse
This script prepares the dataset following proper cross-validation procedures
"""

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import shutil
from tqdm import tqdm
import soundfile as sf

warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = Path('./urban8k_ds')  # Update this to your dataset path
AUDIO_PATH = DATASET_PATH / 'audio'
METADATA_PATH = DATASET_PATH / 'UrbanSound8K.csv'
OUTPUT_PATH = Path('EdgeImpulse_Dataset')
MIXED_TEST_PATH = Path('Mixed_Audio_Test')

# Edge Impulse specifications for audio
TARGET_SR = 16000  # 16kHz is recommended for Edge Impulse
AUDIO_LENGTH = 4.0  # seconds - Edge Impulse works well with fixed-length audio

# Create output directories
OUTPUT_PATH.mkdir(exist_ok=True)
MIXED_TEST_PATH.mkdir(exist_ok=True)

class UrbanSoundPreprocessor:
    def __init__(self, metadata_path, audio_path):
        self.metadata = pd.read_csv(metadata_path)
        self.audio_path = audio_path
        
    def load_audio(self, filename, fold):
        """Load audio file with proper sampling rate"""
        file_path = self.audio_path / f'fold{fold}' / filename
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        return audio, sr
    
    def pad_or_truncate(self, audio, target_length):
        """Ensure all audio clips are the same length"""
        target_samples = int(target_length * TARGET_SR)
        
        if len(audio) > target_samples:
            # Truncate from center
            start = (len(audio) - target_samples) // 2
            audio = audio[start:start + target_samples]
        elif len(audio) < target_samples:
            # Pad with zeros
            pad_width = target_samples - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        
        return audio
    
    def extract_features(self, audio, sr):
        """Extract features for visualization and analysis"""
        features = {}
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Energy
        features['rms_energy'] = np.mean(librosa.feature.rms(y=audio))
        
        return features
    
    def perform_eda(self):
        """Exploratory Data Analysis"""
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        print(f"\nDataset Shape: {self.metadata.shape}")
        print(f"Total Samples: {len(self.metadata)}")
        print(f"\nColumns: {self.metadata.columns.tolist()}")
        
        # Class distribution
        print("\n--- Class Distribution ---")
        class_dist = self.metadata['class'].value_counts().sort_index()
        print(class_dist)
        
        # Fold distribution
        print("\n--- Fold Distribution ---")
        fold_dist = self.metadata['fold'].value_counts().sort_index()
        print(fold_dist)
        
        # Salience distribution
        print("\n--- Salience Distribution ---")
        salience_dist = self.metadata['salience'].value_counts()
        print(salience_dist)
        print("(1 = foreground, 2 = background)")
        
        # Duration statistics
        print("\n--- Duration Statistics ---")
        self.metadata['duration'] = self.metadata['end'] - self.metadata['start']
        print(self.metadata['duration'].describe())
        
        # Check for missing values
        print("\n--- Missing Values ---")
        print(self.metadata.isnull().sum())
        
        # Class distribution per fold
        print("\n--- Class Distribution per Fold ---")
        class_fold = pd.crosstab(self.metadata['fold'], self.metadata['class'])
        print(class_fold)
        
        return self.metadata
    
    def visualize_dataset(self):
        """Create visualizations for the dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        self.metadata['class'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], color='skyblue'
        )
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Sound Class')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Duration distribution
        axes[0, 1].hist(self.metadata['duration'], bins=30, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Audio Duration Distribution')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Count')
        
        # Salience distribution by class
        salience_class = pd.crosstab(self.metadata['class'], self.metadata['salience'])
        salience_class.plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#ff9999', '#66b3ff'])
        axes[1, 0].set_title('Salience Distribution by Class')
        axes[1, 0].set_xlabel('Sound Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(['Foreground', 'Background'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Fold distribution
        self.metadata['fold'].value_counts().sort_index().plot(
            kind='bar', ax=axes[1, 1], color='coral'
        )
        axes[1, 1].set_title('Samples per Fold')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {OUTPUT_PATH / 'dataset_overview.png'}")
        plt.close()
    
    def analyze_audio_samples(self, num_samples=10):
        """Analyze a few audio samples and visualize their characteristics"""
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 3))
        
        # Sample one file from each class
        sampled_files = self.metadata.groupby('class').first().reset_index()
        
        for idx, row in sampled_files.iterrows():
            if idx >= num_samples:
                break
                
            audio, sr = self.load_audio(row['slice_file_name'], row['fold'])
            
            # Waveform
            axes[idx, 0].plot(audio)
            axes[idx, 0].set_title(f"{row['class']} - Waveform")
            axes[idx, 0].set_xlabel('Sample')
            axes[idx, 0].set_ylabel('Amplitude')
            
            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[idx, 1])
            axes[idx, 1].set_title(f"{row['class']} - Spectrogram")
            
            # MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[idx, 2])
            axes[idx, 2].set_title(f"{row['class']} - MFCC")
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'audio_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Audio analysis saved to: {OUTPUT_PATH / 'audio_analysis.png'}")
        plt.close()
    
    def prepare_for_edge_impulse(self, test_fold=10):
        """
        Prepare dataset for Edge Impulse following proper 10-fold CV procedure
        For Edge Impulse, we'll prepare one train/test split at a time
        """
        print("\n" + "=" * 60)
        print(f"PREPARING DATA FOR EDGE IMPULSE - Test Fold: {test_fold}")
        print("=" * 60)
        
        # Create directories for this fold
        fold_output = OUTPUT_PATH / f'fold_{test_fold}_test'
        train_dir = fold_output / 'training'
        test_dir = fold_output / 'testing'
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Split data
        train_data = self.metadata[self.metadata['fold'] != test_fold]
        test_data = self.metadata[self.metadata['fold'] == test_fold]
        
        print(f"\nTraining samples: {len(train_data)}")
        print(f"Testing samples: {len(test_data)}")
        
        # Process training data
        print("\nProcessing training data...")
        train_metadata = []
        for idx, row in tqdm(train_data.iterrows(), total=len(train_data)):
            try:
                audio, sr = self.load_audio(row['slice_file_name'], row['fold'])
                audio = self.pad_or_truncate(audio, AUDIO_LENGTH)
                
                # Create filename with class label for Edge Impulse
                output_filename = f"{row['class']}.{row['slice_file_name']}"
                output_path = train_dir / output_filename
                
                # Save audio file
                sf.write(output_path, audio, TARGET_SR)
                
                train_metadata.append({
                    'filename': output_filename,
                    'class': row['class'],
                    'classID': row['classID'],
                    'original_fold': row['fold'],
                    'salience': row['salience']
                })
            except Exception as e:
                print(f"Error processing {row['slice_file_name']}: {e}")
        
        # Process testing data
        print("\nProcessing testing data...")
        test_metadata = []
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
            try:
                audio, sr = self.load_audio(row['slice_file_name'], row['fold'])
                audio = self.pad_or_truncate(audio, AUDIO_LENGTH)
                
                output_filename = f"{row['class']}.{row['slice_file_name']}"
                output_path = test_dir / output_filename
                
                sf.write(output_path, audio, TARGET_SR)
                
                test_metadata.append({
                    'filename': output_filename,
                    'class': row['class'],
                    'classID': row['classID'],
                    'original_fold': row['fold'],
                    'salience': row['salience']
                })
            except Exception as e:
                print(f"Error processing {row['slice_file_name']}: {e}")
        
        # Save metadata
        train_df = pd.DataFrame(train_metadata)
        test_df = pd.DataFrame(test_metadata)
        
        train_df.to_csv(fold_output / 'train_metadata.csv', index=False)
        test_df.to_csv(fold_output / 'test_metadata.csv', index=False)
        
        print(f"\nData prepared successfully!")
        print(f"Training data: {train_dir}")
        print(f"Testing data: {test_dir}")
        
        return train_df, test_df

    def create_mixed_audio_testset(self, num_samples=100, num_classes_per_mix=2):
        """
        Create multilabel test set by mixing audio from different classes
        This simulates real-world scenarios with multiple sounds occurring simultaneously
        """
        print("\n" + "=" * 60)
        print("CREATING MIXED AUDIO TEST SET")
        print("=" * 60)
        
        mixed_dir = MIXED_TEST_PATH / 'mixed_audio'
        mixed_dir.mkdir(parents=True, exist_ok=True)
        
        classes = self.metadata['class'].unique()
        mixed_metadata = []
        
        print(f"\nCreating {num_samples} mixed audio samples...")
        print(f"Each sample will contain {num_classes_per_mix} different sound classes")
        
        for i in tqdm(range(num_samples)):
            # Randomly select classes to mix
            selected_classes = np.random.choice(classes, size=num_classes_per_mix, replace=False)
            
            mixed_audio = np.zeros(int(AUDIO_LENGTH * TARGET_SR))
            
            # Load and mix audio from each selected class
            labels = []
            for class_name in selected_classes:
                # Get a random sample from this class
                class_samples = self.metadata[self.metadata['class'] == class_name]
                sample = class_samples.sample(1).iloc[0]
                
                # Load audio
                audio, sr = self.load_audio(sample['slice_file_name'], sample['fold'])
                audio = self.pad_or_truncate(audio, AUDIO_LENGTH)
                
                # Mix with varying weights to simulate different volumes
                weight = np.random.uniform(0.4, 1.0)
                mixed_audio += weight * audio
                
                labels.append(class_name)
            
            # Normalize mixed audio
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
            
            # Add slight noise to simulate real-world conditions
            noise = np.random.normal(0, 0.005, len(mixed_audio))
            mixed_audio = mixed_audio + noise
            
            # Save mixed audio
            filename = f"mixed_{i:04d}_{'_'.join(labels)}.wav"
            output_path = mixed_dir / filename
            sf.write(output_path, mixed_audio, TARGET_SR)
            
            mixed_metadata.append({
                'filename': filename,
                'labels': ','.join(labels),
                'num_classes': len(labels),
                'classes': labels
            })
        
        # Save metadata
        mixed_df = pd.DataFrame(mixed_metadata)
        mixed_df.to_csv(MIXED_TEST_PATH / 'mixed_audio_metadata.csv', index=False)
        
        print(f"\nMixed audio test set created successfully!")
        print(f"Location: {mixed_dir}")
        print(f"Metadata: {MIXED_TEST_PATH / 'mixed_audio_metadata.csv'}")
        
        # Statistics
        print("\n--- Mixed Audio Statistics ---")
        print(f"Total mixed samples: {len(mixed_df)}")
        print(f"Classes per sample: {num_classes_per_mix}")
        
        return mixed_df

# Main execution
if __name__ == "__main__":
    print("UrbanSound8K Dataset Preparation for Edge Impulse")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = UrbanSoundPreprocessor(METADATA_PATH, AUDIO_PATH)
    
    # Step 1: EDA
    print("\nStep 1: Exploratory Data Analysis")
    preprocessor.perform_eda()
    
    # Step 2: Visualizations
    print("\nStep 2: Creating Visualizations")
    preprocessor.visualize_dataset()
    preprocessor.analyze_audio_samples()
    
    # Step 3: Prepare data for Edge Impulse
    # Following proper 10-fold CV: prepare one fold as test (e.g., fold 10)
    print("\nStep 3: Preparing Data for Edge Impulse")
    train_df, test_df = preprocessor.prepare_for_edge_impulse(test_fold=10)
    
    # Step 4: Create mixed audio test set
    print("\nStep 4: Creating Mixed Audio Test Set")
    mixed_df = preprocessor.create_mixed_audio_testset(num_samples=100, num_classes_per_mix=2)
    
    # Create summary report
    summary = {
        'dataset': 'UrbanSound8K',
        'target_sample_rate': TARGET_SR,
        'audio_length_seconds': AUDIO_LENGTH,
        'num_classes': len(preprocessor.metadata['class'].unique()),
        'classes': preprocessor.metadata['class'].unique().tolist(),
        'training_samples': len(train_df),
        'testing_samples': len(test_df),
        'mixed_audio_samples': len(mixed_df),
        'folds_used': {
            # .tolist() automatically converts numpy types to python int
            'training': train_df['original_fold'].unique().tolist(), 
            'testing': test_df['original_fold'].unique().tolist()
        }
    }
    
    with open(OUTPUT_PATH / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    