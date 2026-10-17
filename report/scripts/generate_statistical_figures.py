import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure publication-grade styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E2E8F0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Resolve output figures directory
OUT_DIR = os.path.join(repo_root, "report", "figures_statistical")
os.makedirs(OUT_DIR, exist_ok=True)

# Resolve raw CSV path (check simulations subfolder first)
possible_csvs = [
    os.path.join(repo_root, "simulations", "benchmark_1000_runs_raw.csv"),
    os.path.join(script_dir, "..", "..", "simulations", "benchmark_1000_runs_raw.csv"),
    os.path.join("simulations", "benchmark_1000_runs_raw.csv"),
    "benchmark_1000_runs_raw.csv"
]
raw_csv = next((p for p in possible_csvs if os.path.exists(p)), possible_csvs[0])
df_raw = pd.read_csv(raw_csv)

models = [
    ("1. Baseline LogReg", "1. Baseline Logistic Regression", "#3B82F6"),
    ("2. Cost-Sens. LogReg", "2. Cost-Sensitive LogReg", "#EF4444"),
    ("3. Deep MLP", "3. Neural Network (MLP)", "#8B5CF6"),
    ("4. GBDT Classifier", "4. GBDT Classifier", "#10B981"),
    ("5. Isolation Forest", "5. Isolation Forest (Unsupervised)", "#F59E0B"),
    ("6. Soft-Voting Ensemble", "6. Soft-Voting Ensemble", "#2563EB"),
    ("7. Stacked Meta-Learner", "7. Stacked Meta-Learner", "#64748B"),
]

# ==========================================
# FIGURE 1: Net Financial Loss Distribution Boxplots
# ==========================================
fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=300)

loss_data = [df_raw[f"{m_full}__Net_Loss"] for _, m_full, _ in models]
labels = [m_short for m_short, _, _ in models]

bp = ax.boxplot(
    loss_data, patch_artist=True, tick_labels=labels,
    medianprops=dict(color='#0F172A', linewidth=1.5),
    whiskerprops=dict(color='#64748B', linewidth=1.0),
    capprops=dict(color='#64748B', linewidth=1.0),
    flierprops=dict(marker='o', markersize=2, alpha=0.3, markerfacecolor='#94A3B8', markeredgecolor='none')
)

for patch, (_, _, color) in zip(bp['boxes'], models):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
    patch.set_edgecolor('#0F172A')
    patch.set_linewidth(1.0)

# Add Mean markers
means = [d.mean() for d in loss_data]
ax.plot(range(1, len(models) + 1), means, 'D', color='#0F172A', markersize=5, label='Sample Mean ($)')

ax.set_title("Total Net Operational Loss Distribution Across 1,000 Monte Carlo Runs ($N = 1,000$)", fontsize=11, fontweight='bold', color='#0F172A', pad=10)
ax.set_ylabel("Total Operational Loss per Split ($)", fontsize=9, fontweight='bold', color='#1E293B')
ax.grid(True, axis='y')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.xticks(rotation=15, ha='right', fontsize=8, fontweight='normal')
plt.yticks(fontsize=8)

# Annotations
ax.annotate("GBDT Champion\nMean: $4,369\nStd: $1,253", xy=(4, means[3]), xytext=(4.3, 8500),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=1.2),
            fontsize=7.5, fontweight='bold', color='#059669',
            bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#10B981", lw=0.8))

ax.annotate("Cost-Sensitive Alert Explosion\nMean: $21,774\n(+$17,405 loss vs GBDT)", xy=(2, means[1]), xytext=(2.3, 16000),
            arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2),
            fontsize=7.5, fontweight='bold', color='#DC2626',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#EF4444", lw=0.8))

ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig1_loss_distribution_boxplots.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Generated: {fig1_path}")

# ==========================================
# FIGURE 2: The Recall vs. False Positive Explosion (The Recall Fallacy)
# ==========================================
fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)

recalls = [df_raw[f"{m_full}__Recall"].mean() * 100 for _, m_full, _ in models]
fps = [df_raw[f"{m_full}__FP"].mean() for _, m_full, _ in models]

for i, (m_short, _, color) in enumerate(models):
    ax.scatter(recalls[i], fps[i], color=color, s=120, edgecolors='#0F172A', linewidth=1.2, zorder=5)
    offset_y = 180 if m_short != "2. Cost-Sens. LogReg" else -450
    offset_x = 0.3 if m_short != "2. Cost-Sens. LogReg" else -2.5
    ax.annotate(f"{m_short}\n({recalls[i]:.1f}%, {fps[i]:.0f} FPs)",
                xy=(recalls[i], fps[i]), xytext=(recalls[i] + offset_x, fps[i] + offset_y),
                fontsize=7.5, fontweight='bold', color='#1E293B',
                bbox=dict(boxstyle="round,pad=0.2", fc="#FFFFFF", ec="#CBD5E1", alpha=0.85))

# Plot exponential trajectory
sorted_indices = np.argsort(recalls)
ax.plot(np.array(recalls)[sorted_indices], np.array(fps)[sorted_indices], '--', color='#94A3B8', alpha=0.7, zorder=2)

ax.set_title("The 'Recall Fallacy': Chasing the Final 7% Recall Triggers an 860% Alert Explosion", fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
ax.set_xlabel("Event Recall (%) — Security Catch Rate", fontsize=9, fontweight='bold', color='#1E293B')
ax.set_ylabel("Mean False Alarms (FP Count per Test Split)", fontsize=9, fontweight='bold', color='#1E293B')
ax.grid(True)
ax.set_xlim(55, 100)
ax.set_ylim(0, 9500)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# Shaded Operational Zone
ax.axhspan(0, 2000, color='#ECFDF5', alpha=0.5, label='Manageable Analyst Triage Zone (FPR < 3.5%)')
ax.axhspan(2000, 9500, color='#FEF2F2', alpha=0.4, label='Severe Queue Overload Zone (FPR > 3.5%)')
ax.legend(loc='upper left', fontsize=7.5, framealpha=0.95)

plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "fig2_recall_vs_false_alarms.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Generated: {fig2_path}")

# ==========================================
# FIGURE 3: Operational Cost Sensitivity Curves & Breakeven Frontier
# ==========================================
fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=300)

c_vals = np.linspace(0.10, 5.00, 100)

mean_fns = {
    "Baseline LogReg": 1770.57,
    "Cost-Sens. LogReg": 663.85,
    "Neural Net (MLP)": 1778.08,
    "GBDT Classifier": 2168.80,
    "Isolation Forest": 5066.98,
    "Soft-Voting Ensemble": 1590.85,
    "Stacked Meta-Learner": 1464.31,
}

mean_fps = {
    "Baseline LogReg": 1420.592,
    "Cost-Sens. LogReg": 8444.065,
    "Neural Net (MLP)": 1485.818,
    "GBDT Classifier": 880.185,
    "Isolation Forest": 688.754,
    "Soft-Voting Ensemble": 1789.139,
    "Stacked Meta-Learner": 2421.549,
}

colors_dict = {
    "Baseline LogReg": "#3B82F6",
    "Cost-Sens. LogReg": "#EF4444",
    "Neural Net (MLP)": "#8B5CF6",
    "GBDT Classifier": "#10B981",
    "Isolation Forest": "#F59E0B",
    "Soft-Voting Ensemble": "#2563EB",
    "Stacked Meta-Learner": "#64748B",
}

for name in mean_fns.keys():
    loss_curve = mean_fns[name] + (mean_fps[name] * c_vals)
    lw = 2.4 if name in ["GBDT Classifier", "Soft-Voting Ensemble", "Cost-Sens. LogReg"] else 1.2
    ls = '-' if name in ["GBDT Classifier", "Soft-Voting Ensemble"] else ('--' if name == "Cost-Sens. LogReg" else ':')
    ax.plot(c_vals, loss_curve, label=name, color=colors_dict[name], linewidth=lw, linestyle=ls)

# Breakeven point: $0.6358
c_star = 0.6358
loss_star = 2168.80 + (880.185 * c_star)

ax.plot(c_star, loss_star, 'o', color='#DC2626', markersize=7, zorder=6)
ax.axvline(c_star, color='#DC2626', linestyle='--', linewidth=1.0, alpha=0.8)

# Shaded domains
ax.axvspan(0.10, c_star, color='#EFF6FF', alpha=0.4, label='Domain 1: Soft-Voting Optimal (Push Challenges)')
ax.axvspan(c_star, 5.00, color='#ECFDF5', alpha=0.4, label='Domain 2: GBDT Optimal (Standard Manual Triage)')

ax.annotate(f"Breakeven Point\n$C_{{FP}}^* = \\$0.6358$\nLoss = ${loss_star:.0f}",
            xy=(c_star, loss_star), xytext=(c_star + 0.35, loss_star + 1800),
            arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2),
            fontsize=8, fontweight='bold', color='#DC2626',
            bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", lw=0.8))

ax.set_title("Operational Cost Sensitivity Frontier: Net Loss ($) vs. Review Overhead ($C_{FP}$)", fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
ax.set_xlabel("Cost per False Alarm Review ($C_{FP}$) [USD]", fontsize=9, fontweight='bold', color='#1E293B')
ax.set_ylabel("Total Operational Net Loss ($)", fontsize=9, fontweight='bold', color='#1E293B')
ax.set_xlim(0.10, 5.00)
ax.set_ylim(1000, 26000)
ax.grid(True)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:.2f}'))
ax.legend(loc='upper left', fontsize=7.0, framealpha=0.92, ncol=2)

plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "fig3_breakeven_cost_curves.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"Generated: {fig3_path}")

# ==========================================
# FIGURE 4: Extreme Tail Risk Exposure (VaR 99% vs. CVaR 99%)
# ==========================================
fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=300)

models_risk = [
    "1. Baseline LogReg",
    "2. Cost-Sens. LogReg",
    "3. Neural Net (MLP)",
    "4. GBDT Classifier",
    "5. Isolation Forest",
    "6. Soft-Voting Ensemble",
    "7. Stacked Meta-Learner"
]

var_99 = [8308.31, 26449.77, 8624.46, 7622.60, 11741.74, 9174.48, 10673.71]
cvar_99 = [8934.98, 27284.65, 8942.64, 8117.52, 12969.31, 9396.16, 10885.98]

x = np.arange(len(models_risk))
width = 0.35

rects1 = ax.bar(x - width/2, var_99, width, label='99% Value-at-Risk (VaR 99%)', color='#93C5FD', edgecolor='#1E3A8A', lw=0.8)
rects2 = ax.bar(x + width/2, cvar_99, width, label='99% Conditional VaR / Expected Shortfall (CVaR 99%)', color='#1E3A8A', edgecolor='#0F172A', lw=0.8)

# Highlight GBDT bars
rects1[3].set_color('#6EE7B7')
rects1[3].set_edgecolor('#065F46')
rects2[3].set_color('#059669')
rects2[3].set_edgecolor('#064E3B')

# Highlight Cost-Sensitive bars
rects1[1].set_color('#FCA5A5')
rects1[1].set_edgecolor('#991B1B')
rects2[1].set_color('#DC2626')
rects2[1].set_edgecolor('#7F1D1D')

ax.set_title("Institutional Tail Risk Profile: Capital Loss in Worst 1% Disaster Scenarios", fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
ax.set_ylabel("Capital at Risk in Tail ($)", fontsize=9, fontweight='bold', color='#1E293B')
ax.set_xticks(x)
ax.set_xticklabels(models_risk, rotation=15, ha='right', fontsize=7.5, fontweight='normal')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax.grid(True, axis='y')
ax.set_ylim(0, 31000)

# Annotate GBDT
ax.annotate("Lowest Tail Exposure\nCVaR = $8,118", xy=(3 + width/2, 8117.52), xytext=(3 + width/2 + 0.3, 12500),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=1.2),
            fontsize=7.5, fontweight='bold', color='#059669',
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#10B981", lw=0.8))

# Annotate Cost-Sensitive
ax.annotate("Catastrophic Exposure\nCVaR = $27,285", xy=(1 + width/2, 27284.65), xytext=(1 - 0.6, 28500),
            arrowprops=dict(arrowstyle="->", color='#DC2626', lw=1.2),
            fontsize=7.5, fontweight='bold', color='#DC2626',
            bbox=dict(boxstyle="round,pad=0.25", fc="#FEF2F2", ec="#EF4444", lw=0.8))

ax.legend(loc='upper right', fontsize=8, framealpha=0.92)
plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "fig4_tail_risk_var_cvar.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"Generated: {fig4_path}")

# ==========================================
# FIGURE 5: Fraud Investigator's Workday Breakdown (Precision vs. FDR)
# ==========================================
fig, ax = plt.subplots(figsize=(8.0, 3.8), dpi=300)

precisions = [5.95, 1.11, 5.72, 9.30, 8.22, 4.82, 3.63]
fdrs = [100.0 - p for p in precisions]

y_pos = np.arange(len(models_risk))

ax.barh(y_pos, precisions, color='#10B981', edgecolor='#065F46', height=0.55, label='Actual Fraudsters Caught (Precision %)')
ax.barh(y_pos, fdrs, left=precisions, color='#F1F5F9', edgecolor='#94A3B8', height=0.55, label='Innocent Cardholders Disrupted (FDR %)')

# Highlight GBDT
ax.barh(3, precisions[3], color='#059669', edgecolor='#064E3B', height=0.55)
# Highlight Cost-Sensitive
ax.barh(1, precisions[1], color='#EF4444', edgecolor='#B91C1C', height=0.55)

for i in range(len(models_risk)):
    ax.text(precisions[i] + 1.0, y_pos[i], f"{precisions[i]:.1f}% Real ({precisions[i]:.0f}/100)",
            va='center', fontsize=7.2, fontweight='bold', color='#1E293B')

ax.set_title("The Fraud Investigator's Workday: What Happens to 100 Alerts Sent to Triage?", fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
ax.set_xlabel("Queue Composition Percentage (%)", fontsize=9, fontweight='bold', color='#1E293B')
ax.set_yticks(y_pos)
ax.set_yticklabels(models_risk, fontsize=8, fontweight='normal')
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(ticker.PercentFormatter())
ax.grid(True, axis='x')
ax.legend(loc='lower right', fontsize=8, framealpha=0.95)

plt.tight_layout()
fig5_path = os.path.join(OUT_DIR, "fig5_investigator_workday_efficiency.png")
plt.savefig(fig5_path, dpi=300)
plt.close()
print(f"Generated: {fig5_path}")

print("All 5 statistical figures successfully generated!")
