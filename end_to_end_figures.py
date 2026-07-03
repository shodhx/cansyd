import matplotlib.pyplot as plt
import numpy as np

from cnsd.datasets.xjtusy import load_xjtusy_domain_split
from cnsd.diagnosis.system import CNSD

print('Loading XJTU-SY...')
train_ds, target_ds = load_xjtusy_domain_split(window_size=4096)

# Find a severe inner race fault (class > 0, likely inner race is 1 or 2)
fault_indices = np.where(target_ds.y > 0)[0]
# Let's take a specific index
idx = fault_indices[15]
raw_signal = target_ds.X[idx].flatten()
condition = target_ds.cond[idx]
true_label = target_ds.y[idx]

print(f'Selected Trajectory: Index {idx}, Condition {condition}Hz, True Label {true_label}')

# ---------------------------------------------------------
# LAYER 1: PERCEPTION
# ---------------------------------------------------------
model = CNSD(conf_thresh=0.90)
print('Training quick CNN for Layer 1...')
model.fit(train_ds, epochs=1)  # Just need a trained weights struct

probs = model.cnn.predict(target_ds.X[idx : idx + 1], verbose=0)[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(raw_signal, color='blue', alpha=0.7)
ax1.set_title(f'Layer 1 (Perception) - Raw Vibration Signal (Condition={condition}Hz)')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Amplitude')

classes = [f'Class {i}' for i in range(len(probs))]
ax2.bar(classes, probs, color=['green' if i == 0 else 'red' for i in range(len(probs))])
ax2.set_title('CNN Fault Probabilities')
ax2.set_ylabel('Probability')
ax2.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig('paper_layer1_perception.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved paper_layer1_perception.png')

# ---------------------------------------------------------
# LAYER 2: PHYSICS VERIFICATION
# ---------------------------------------------------------
diag = model.symbolic.diagnose(raw_signal, probs.argmax(), condition)

# We need to manually plot the envelope spectrum for the figure
from cnsd.physics.bearing import envelope_spectrum  # noqa: E402

freqs, mag = envelope_spectrum(raw_signal, fs=target_ds.fs)

plt.figure(figsize=(8, 4))
plt.plot(freqs, mag, color='purple', alpha=0.7)
plt.title('Layer 2: Envelope Spectrum (Physics Verification)', fontsize=12)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 500)  # Zoom into the interesting region

# Add red vertical lines for the fault frequencies if they exist
# Fallback if explanation is a dict (just in case), otherwise skip lines
if isinstance(diag.get('explanation'), dict):
    for key, freq in diag['explanation'].get('freqs_hz', {}).items():
        plt.axvline(freq, color='red', linestyle='--', alpha=0.5, label=f'{key}: {freq:.1f}Hz')

handles, labels = plt.gca().get_legend_handles_labels()
if handles:
    plt.legend()
plt.tight_layout()
plt.savefig('paper_layer2_physics.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved paper_layer2_physics.png')

# ---------------------------------------------------------
# LAYER 5: DECISION
# ---------------------------------------------------------
# Layer 3 and 4 are handled by the other script. Layer 5 outputs the action
from cnsd.consensus import fuse  # noqa: E402

status = fuse(diag['verdict'], probs.max(), model.conf_thresh)

print('\n--- FINAL LAYER 5 ACTION ---')
print(f'Status: {status}')
print(f'Action: {diag["action"]}')
