import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import math

# --- 1. Global Visual Configuration ---
plt.rcParams.update({
    'font.size': 22,                
    'axes.titlesize': 28,           
    'axes.labelsize': 26,           
    'xtick.labelsize': 22,          
    'ytick.labelsize': 22,          
    'legend.fontsize': 22,          
    'axes.titleweight': 'bold',     
    'axes.linewidth': 2.5,          
    'figure.titlesize': 32          
})

FILES = {
    "Training Data": "unique_combined_sequences.txt",
    "Finetuned Model": "amp_samples_finetuned_code_1.txt",
    "Pretrained Model": "amp_samples_pretrained_code_1.txt",
}

# Kyte-Doolittle Scale
KD_SCALE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def calculate_hm(sequence, angle_deg):
    theta = math.radians(angle_deg)
    sum_cos = sum_sin = 0.0
    for i, aa in enumerate(sequence):
        h = KD_SCALE.get(aa, 0.0)
        sum_cos += h * math.cos(i * theta)
        sum_sin += h * math.sin(i * theta)
    return math.sqrt(sum_cos**2 + sum_sin**2) / len(sequence)

def clean_and_analyze(seq):
    replacements = {'U':'C', 'Z':'E', 'B':'D', 'X':'A', 'J':'L', 'O':'K'}
    for non_std, std in replacements.items():
        seq = seq.replace(non_std, std)
    if len(seq) == 0: return None
    analysis = ProteinAnalysis(seq)
    res = {
        "Sequence": seq,
        "Length": len(seq),
        "GRAVY": analysis.gravy(),
        "Hydrophobic Moment\n(Helix)": calculate_hm(seq, 100),
        "Hydrophobic Moment\n(Sheet)": calculate_hm(seq, 175)
    }
    try:
        res["Isoelectric Point"] = analysis.isoelectric_point()
    except:
        res["Isoelectric Point"] = np.nan
    return res

if __name__ == "__main__":
    all_physico_data = []
    overall_composition = {}

    for label, path in FILES.items():
        print(f"Processing {label}...")
        try:
            with open(path, "r") as f:
                sequences = [line.strip().upper() for line in f if line.strip()]
            file_combined_text = ""
            for s in sequences:
                props = clean_and_analyze(s)
                if props:
                    props["Source"] = label
                    all_physico_data.append(props)
                    file_combined_text += props["Sequence"]
            counts = Counter(file_combined_text)
            total_aa = sum(counts.values())
            overall_composition[label] = {aa: (counts[aa]/total_aa)*100 if total_aa > 0 else 0 for aa in AMINO_ACIDS}
        except Exception as e:
            print(f"Error: {e}")

    df_physico = pd.DataFrame(all_physico_data)
    df_comp = pd.DataFrame(overall_composition)
    sources = list(FILES.keys())
    colors = ['#4C72B0', '#DD8452', '#55A868']

    # --- Plot 1: Physicochemical Grid (2 Rows x 3 Columns) ---
    PROPS = ["Isoelectric Point", "GRAVY", "Length", "Hydrophobic Moment\n(Helix)", "Hydrophobic Moment\n(Sheet)"]
    
    # 2 rows, 3 columns = 6 slots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
    axes_flat = axes.flatten()

    for i, prop in enumerate(PROPS):
        data = [df_physico[(df_physico['Source'] == s) & (df_physico[prop].notna())][prop].values for s in sources]
        parts = axes_flat[i].violinplot(data, showmedians=True)
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color); pc.set_alpha(0.6); pc.set_edgecolor('black'); pc.set_linewidth(2)
        for l in ('cbars','cmins','cmaxes','cmedians'):
            parts[l].set_edgecolor('black'); parts[l].set_linewidth(3)

        axes_flat[i].set_title(prop, pad=20)
        axes_flat[i].set_xticks(range(1, len(sources) + 1))
        axes_flat[i].set_xticklabels(sources, rotation=15, ha='right')
        axes_flat[i].set_ylabel("Value", labelpad=15, fontweight='bold')
        axes_flat[i].grid(axis='y', linestyle='--', alpha=0.4)

    # Hide the 6th empty slot (bottom right)
    for j in range(len(PROPS), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.savefig("amp_physicochemical_violins_2x3.png", dpi=300)
    print("Saved: amp_physicochemical_violins_2x3.png")

    # --- Plot 2: Individual AA Composition ---
    plt.figure(figsize=(16, 9))
    df_comp.plot(kind='bar', width=0.8, color=colors, ax=plt.gca(), edgecolor='black', linewidth=1.5)
#    plt.title("Individual Amino Acid Composition (%)", pad=30)
    plt.ylabel("Percent (%)", labelpad=20, fontweight='bold'); plt.xlabel("Amino Acid", labelpad=20, fontweight='bold')
    plt.xticks(rotation=0); plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig("amp_individual_composition_bar.png", dpi=300)

    # --- Plot 3: Grouped Chemical Properties ---
    GROUPS = {
        "Positively Charged (KRH)": list("KRH"),
        "Negatively Charged (DE)": list("DE"),
        "Nonpolar (AVLIFMWP)": list("AVLIFMWP"),
        "Polar Uncharged (STCNQ)": list("STCNQ"),
        "Structural/Special (PCG)": list("PCG")
    }
    group_results = {g: df_comp.loc[aa].sum() for g, aa in GROUPS.items()}
    df_grouped = pd.DataFrame(group_results).T

    plt.figure(figsize=(16, 9))
    df_grouped.plot(kind='bar', width=0.8, color=colors, ax=plt.gca(), edgecolor='black', linewidth=2.5)
#    plt.title("Grouped Amino Acid Composition", pad=40)
    plt.ylabel("Cumulative Percent (%)", labelpad=25, fontweight='bold'); plt.xlabel("Chemical Grouping", labelpad=25, fontweight='bold')
    plt.xticks(rotation=15, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.savefig("amp_grouped_properties_bar.png", dpi=300)
    print("Saved: amp_grouped_properties_bar.png")
