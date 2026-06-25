"""
MiniHack: AI Data Intelligence Challenge
Complete Solution on the REAL Last Mile Delivery Dataset
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Colour Palette ───────────────────────────────────────────────────────────
C = {
    'blue':    '#2563EB', 'purple':  '#7C3AED', 'green':   '#16A34A',
    'amber':   '#D97706', 'red':     '#DC2626',  'teal':    '#0891B2',
    'pink':    '#DB2777', 'bg':      '#F1F5F9',  'card':    '#FFFFFF',
    'text':    '#1E293B', 'muted':   '#64748B',
}
plt.rcParams.update({
    'figure.facecolor': C['bg'], 'axes.facecolor': C['card'],
    'axes.edgecolor': '#CBD5E1', 'axes.labelcolor': C['text'],
    'xtick.color': C['text'],    'ytick.color': C['text'],
    'text.color': C['text'],     'font.family': 'DejaVu Sans',
    'axes.spines.top': False,    'axes.spines.right': False,
    'axes.grid': True,           'grid.alpha': 0.4,
    'grid.linestyle': '--',
})

# ══════════════════════════════════════════════════════════════════════════════
# 0 — LOAD + CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("="*65)
print("  STEP 0 — DATA LOADING & CLEANING")
print("="*65)

raw = pd.read_csv('/mnt/user-data/uploads/last_mile_delivery_dataset.csv')
df  = raw.copy()
print(f"Raw shape: {df.shape}")

# --- Fix dirty city names ---
city_map = {
    ' ahmedabad':'Ahmedabad','ahmedabad':'Ahmedabad',
    'bangaluru':'Bangalore','bangalore':'Bangalore',
    'chennai':'Chennai',
    'delhi':'Delhi',
    'hydrabad':'Hyderabad','hyderabad':'Hyderabad',
    'jaipur':'Jaipur','kolkata':'Kolkata','kolkata ':'Kolkata',
    'lucknow':'Lucknow','mumbai':'Mumbai','MUMBAI':'Mumbai','pune':'Pune',
}
df['city'] = df['city'].str.strip().str.lower().map(
    lambda x: city_map.get(x, x.title()))

# --- Standardise vehicle_type ---
df['vehicle_type'] = df['vehicle_type'].str.strip().str.title()
df['vehicle_type'] = df['vehicle_type'].replace({'Auto':'Auto', 'Bike':'Bike',
                                                 'Cycle':'Cycle','Van':'Van'})

# --- Parse time → hour ---
df['hour'] = pd.to_datetime(df['order_time'], format='%H:%M').dt.hour

# --- Parse date ---
df['order_date'] = pd.to_datetime(df['order_date'])
df['month']      = df['order_date'].dt.month
df['month_name'] = df['order_date'].dt.strftime('%b')

# --- Handle negative delay_mins (early deliveries → 0 for analysis) ---
print(f"\nNegative delay_mins: {(df['delay_mins']<0).sum()} rows "
      f"(min={df['delay_mins'].min():.1f}) — treated as early (0 delay)")
df['delay_mins_raw'] = df['delay_mins'].copy()
df['delay_mins'] = df['delay_mins'].clip(lower=0)

# --- Remove outliers (IQR×3) ---
Q1, Q3 = df['delay_mins'].quantile([0.25, 0.75])
IQR = Q3 - Q1
upper = Q3 + 3*IQR
pre = len(df)
df_clean = df[df['delay_mins'] <= upper].copy()
print(f"Outliers removed  : {pre - len(df_clean)} rows (>{upper:.1f} min)")
print(f"Clean shape       : {df_clean.shape}")
print(f"Remaining nulls   : {df_clean.isnull().sum().sum()} (in zone/rider_id/gps — not needed for analysis)")

# --- Derived columns ---
df_clean['is_peak'] = df_clean['hour'].apply(
    lambda h: 'Peak' if (8<=h<=10) or (17<=h<=20) else 'Off-Peak')
df_clean['exp_group'] = pd.cut(df_clean['rider_experience_yrs'],
    bins=[0,2,4,100], labels=['<2 yrs','2–4 yrs','>4 yrs'])
df_clean['on_time_flag'] = (df_clean['delivery_status']=='On-Time').astype(int)

# Show cleaned uniques
print(f"\nCities after cleaning  : {sorted(df_clean['city'].unique())}")
print(f"Vehicles after cleaning: {sorted(df_clean['vehicle_type'].unique())}")

# ══════════════════════════════════════════════════════════════════════════════
# Q1 — PEAK HOUR DELAY PATTERN
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  Q1 — PEAK HOUR DELAY PATTERN")
print("="*65)

grp = df_clean.groupby('is_peak')['delay_mins'].agg(
    Mean='mean', Median='median', Std='std', Count='count').round(2)
print(grp.to_string())

peak_m   = grp.loc['Peak','Mean']
off_m    = grp.loc['Off-Peak','Mean']
diff     = peak_m - off_m
pct_diff = diff / off_m * 100

t, p = stats.ttest_ind(
    df_clean[df_clean['is_peak']=='Peak']['delay_mins'],
    df_clean[df_clean['is_peak']=='Off-Peak']['delay_mins'])
print(f"\n  Peak mean    : {peak_m:.2f} mins")
print(f"  Off-Peak mean: {off_m:.2f} mins")
print(f"  Difference   : +{diff:.2f} mins  ({pct_diff:.1f}% higher during peak)")
print(f"  T-test       : t={t:.3f}, p={p:.2e}  → {'SIGNIFICANT ✓' if p<0.05 else 'not significant'}")

hourly = df_clean.groupby('hour')['delay_mins'].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(C['bg'])
fig.suptitle('Q1 — Peak Hour Delay Analysis (Real Data)',
             fontsize=15, fontweight='bold', y=1.02)

# Bar: peak vs off-peak
ax = axes[0]
bar_c = [C['red'] if 'Peak'==g else C['green'] for g in grp.index]
bars = ax.bar(grp.index, grp['Mean'], color=bar_c, width=0.45,
              edgecolor='white', linewidth=1.5)
for b, v in zip(bars, grp['Mean']):
    ax.text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f} min',
            ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Delay (mins)', fontsize=11)
ax.set_title('Peak vs Off-Peak Mean Delay', fontsize=12, fontweight='bold')
sig = f'Δ = +{diff:.1f} min\n({pct_diff:.0f}% more)\np = {p:.0e}'
ax.text(0.97, 0.97, sig, transform=ax.transAxes, ha='right', va='top',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='#FEF9C3', ec=C['amber']))

# Line: hourly average
ax2 = axes[1]
ax2.plot(hourly['hour'], hourly['delay_mins'], color=C['blue'],
         linewidth=2.5, marker='o', markersize=5, zorder=5)
ax2.fill_between(hourly['hour'], hourly['delay_mins'], alpha=0.1, color=C['blue'])
for span in [(8,10),(17,20)]:
    ax2.axvspan(span[0], span[1], alpha=0.18, color=C['red'])
ax2.set_xlabel('Hour of Day', fontsize=11)
ax2.set_ylabel('Mean Delay (mins)', fontsize=11)
ax2.set_title('Average Delay by Hour', fontsize=12, fontweight='bold')
ax2.set_xticks(range(0, 24, 2))
ax2.legend([plt.Rectangle((0,0),1,1,fc=C['red'],alpha=0.3)],
           ['Peak Windows (8-10am, 5-8pm)'], fontsize=9)
plt.tight_layout()
plt.savefig('/home/claude/q1.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("  [✓] Q1 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Q2 — WEATHER vs DELAY CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  Q2 — WEATHER vs DELAY CORRELATION")
print("="*65)

w_order = ['Clear','Partly Cloudy','Rain','Fog']
w_med = df_clean.groupby('weather_condition')['delay_mins'].median().reindex(w_order)
print(f"  Median delay by weather:\n{w_med.round(2).to_string()}")

cross = df_clean.groupby(['weather_condition','order_type'])['delay_mins'].median().unstack()
rain_row = cross.loc['Rain'] if 'Rain' in cross.index else cross.iloc[0]
hardest = rain_row.idxmax()
print(f"\n  Hardest hit order type in Rain: {hardest} ({rain_row[hardest]:.1f} min median)")
print(cross.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(C['bg'])
fig.suptitle('Q2 — Weather vs Delay Correlation (Real Data)',
             fontsize=15, fontweight='bold')

# Box plot
ax = axes[0]
w_colors = [C['green'], C['teal'], C['blue'], C['purple']]
data_w = [df_clean[df_clean['weather_condition']==w]['delay_mins'].values
          for w in w_order if w in df_clean['weather_condition'].unique()]
labels_w = [w for w in w_order if w in df_clean['weather_condition'].unique()]
bp = ax.boxplot(data_w, labels=labels_w, patch_artist=True,
                medianprops=dict(color='white', linewidth=2.5))
for patch, col in zip(bp['boxes'], w_colors):
    patch.set_facecolor(col); patch.set_alpha(0.72)
for part in ['whiskers','caps']:
    for item in bp[part]: item.set_color(C['muted']); item.set_alpha(0.6)
for fl in bp['fliers']: fl.set(marker='o', alpha=0.2, ms=3, color=C['muted'])
meds = [df_clean[df_clean['weather_condition']==w]['delay_mins'].median()
        for w in labels_w]
for i, med in enumerate(meds):
    ax.text(i+1, med+0.6, f'{med:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('Delay Distribution by Weather', fontsize=12, fontweight='bold')
ax.set_ylabel('Delay (mins)', fontsize=11)
ax.set_xticklabels(labels_w, fontsize=9)

# Heatmap: weather × order_type
ax2 = axes[1]
plot_cross = cross.reindex(labels_w)
im = ax2.imshow(plot_cross.values, aspect='auto', cmap='YlOrRd')
ax2.set_xticks(range(len(plot_cross.columns)))
ax2.set_xticklabels(plot_cross.columns, rotation=30, ha='right', fontsize=9)
ax2.set_yticks(range(len(labels_w)))
ax2.set_yticklabels(labels_w, fontsize=10)
ax2.set_title('Median Delay: Weather × Order Type (mins)', fontsize=12, fontweight='bold')
vmin, vmax = plot_cross.values.min(), plot_cross.values.max()
for i in range(len(labels_w)):
    for j in range(len(plot_cross.columns)):
        val = plot_cross.values[i,j]
        if not np.isnan(val):
            color = 'white' if val > (vmin+vmax)/2 else 'black'
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=9, fontweight='bold', color=color)
plt.colorbar(im, ax=ax2, label='Median Delay (mins)', shrink=0.8)
plt.tight_layout()
plt.savefig('/home/claude/q2.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("  [✓] Q2 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Q3 — RIDER EXPERIENCE EFFECT (Statistics)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  Q3 — RIDER EXPERIENCE EFFECT")
print("="*65)

g1   = df_clean[df_clean['rider_experience_yrs'] <  2]['delay_mins']
gmid = df_clean[(df_clean['rider_experience_yrs'] >= 2) &
                (df_clean['rider_experience_yrs'] <= 4)]['delay_mins']
g2   = df_clean[df_clean['rider_experience_yrs'] >  4]['delay_mins']

for label, g in [('<2 yrs', g1), ('2-4 yrs', gmid), ('>4 yrs', g2)]:
    print(f"  {label:8s}: n={len(g):4d}, mean={g.mean():.2f}, median={g.median():.2f}, std={g.std():.2f}")

t, p    = stats.ttest_ind(g1, g2, equal_var=False)   # Welch's
u, pmwu = stats.mannwhitneyu(g1, g2, alternative='two-sided')
n1, n2  = len(g1), len(g2)
d       = (g1.mean()-g2.mean()) / np.sqrt(((n1-1)*g1.std()**2+(n2-1)*g2.std()**2)/(n1+n2-2))
size    = 'Large' if abs(d)>0.8 else 'Medium' if abs(d)>0.5 else 'Small'

print(f"\n  Welch's t-test : t={t:.3f}, p={p:.2e}")
print(f"  Mann-Whitney U : U={u:.0f}, p={pmwu:.2e}")
print(f"  Cohen's d      : {d:.3f} ({size} effect)")
print(f"  Gap            : {g1.mean()-g2.mean():.2f} mins (<2 yrs vs >4 yrs)")
print(f"  Verdict        : {'SIGNIFICANT ✓' if p<0.05 else 'NOT SIGNIFICANT'} (α=0.05)")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(C['bg'])
fig.suptitle('Q3 — Rider Experience Effect on Delay (Real Data)',
             fontsize=15, fontweight='bold')

# Violin
ax = axes[0]
vdata  = [g1.values, gmid.values, g2.values]
vlabels= ['< 2 yrs\n(Novice)', '2–4 yrs\n(Mid)', '> 4 yrs\n(Expert)']
vp = ax.violinplot(vdata, positions=[1,2,3], showmedians=True)
vcols = [C['red'], C['amber'], C['green']]
for body, col in zip(vp['bodies'], vcols):
    body.set_facecolor(col); body.set_alpha(0.65)
vp['cmedians'].set_color('white'); vp['cmedians'].set_linewidth(2.5)
for part in ['cbars','cmins','cmaxes']:
    vp[part].set_color(C['muted']); vp[part].set_alpha(0.5)
ax.set_xticks([1,2,3])
ax.set_xticklabels(vlabels, fontsize=10)
ax.set_ylabel('Delay (mins)', fontsize=11)
ax.set_title('Delay by Experience Group', fontsize=12, fontweight='bold')
for pos, g in zip([1,2,3],[g1,gmid,g2]):
    ax.text(pos, g.mean()+0.5, f'{g.mean():.1f}', ha='center',
            va='bottom', fontsize=10, fontweight='bold')
sig_txt = f'p = {p:.2e}\nCohen\'s d = {d:.2f} ({size})\nGap = {g1.mean()-g2.mean():.1f} min'
ax.text(0.98, 0.97, sig_txt, transform=ax.transAxes, ha='right', va='top',
        fontsize=9, bbox=dict(boxstyle='round,pad=0.4', fc='#FEF9C3', ec=C['amber']))

# Scatter with regression
ax2 = axes[1]
sample = df_clean.sample(min(600,len(df_clean)), random_state=42)
sc = ax2.scatter(sample['rider_experience_yrs'], sample['delay_mins'],
                 c=sample['delay_mins'], cmap='RdYlGn_r',
                 alpha=0.45, s=22, edgecolors='none')
xv = df_clean['rider_experience_yrs'].astype(float).values
yv = df_clean['delay_mins'].astype(float).values
mask = np.isfinite(xv) & np.isfinite(yv)
if mask.sum() > 2:
    z  = np.polyfit(xv[mask], yv[mask], 1)
    xl = np.linspace(xv[mask].min(), xv[mask].max(), 100)
    ax2.plot(xl, np.poly1d(z)(xl), color=C['red'], linewidth=2.5,
             linestyle='--', label=f'Trend (slope={z[0]:.2f})')
ax2.axvline(2, color=C['amber'], linestyle=':', linewidth=1.5, alpha=0.8)
ax2.axvline(4, color=C['green'], linestyle=':', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Rider Experience (years)', fontsize=11)
ax2.set_ylabel('Delay (mins)', fontsize=11)
ax2.set_title('Experience vs Delay (Scatter + Trend)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
plt.colorbar(sc, ax=ax2, label='Delay (mins)', shrink=0.8)
plt.tight_layout()
plt.savefig('/home/claude/q3.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("  [✓] Q3 saved")

# ══════════════════════════════════════════════════════════════════════════════
# Q4 — CITY-LEVEL PERFORMANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  Q4 — CITY-LEVEL PERFORMANCE DASHBOARD")
print("="*65)

city_ontime  = (df_clean.groupby('city')['on_time_flag'].mean()*100).round(1).sort_values()
monthly_avg  = df_clean.groupby('month')['delay_mins'].mean().reindex(range(1,13))
month_names  = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
vehicle_perf = df_clean.groupby('vehicle_type').agg(
    avg_delay   =('delay_mins','mean'),
    on_time_pct =('on_time_flag','mean')
).round(3)
vehicle_perf['on_time_pct'] *= 100

print(f"\n  City on-time %:\n{city_ontime.to_string()}")
print(f"\n  Vehicle perf:\n{vehicle_perf.round(2).to_string()}")
worst_city    = city_ontime.idxmin()
best_city     = city_ontime.idxmax()
worst_vehicle = vehicle_perf['avg_delay'].idxmax()
peak_month_i  = monthly_avg.idxmax()
peak_month_nm = month_names[peak_month_i-1]

# Build dashboard
fig = plt.figure(figsize=(20, 17))
fig.patch.set_facecolor(C['bg'])
fig.suptitle('Q4 — City-Level Performance Dashboard\nLast Mile Delivery Analytics (Real Data)',
             fontsize=17, fontweight='bold', color=C['text'], y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.50, wspace=0.30,
                       top=0.93, bottom=0.05, left=0.07, right=0.97)

# ── Panel 1: City on-time bar ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
bar_cols = [C['red'] if v<50 else C['amber'] if v<65 else C['green']
            for v in city_ontime.values]
bars1 = ax1.barh(city_ontime.index, city_ontime.values,
                 color=bar_cols, height=0.6, edgecolor='white', linewidth=1.2)
ax1.axvline(50, color='grey', linestyle='--', linewidth=1.5, alpha=0.7)
for b, v in zip(bars1, city_ontime.values):
    ax1.text(v+0.5, b.get_y()+b.get_height()/2,
             f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')
ax1.set_xlabel('On-Time Rate (%)', fontsize=11)
ax1.set_title('Panel 1 — City-Wise On-Time Delivery Rate', fontsize=13, fontweight='bold', pad=10)
ax1.set_xlim(0, 108)
ax1.legend(handles=[Patch(fc=C['red'],label='Poor (<50%)'),
                    Patch(fc=C['amber'],label='Average (50–65%)'),
                    Patch(fc=C['green'],label='Good (>65%)')],
           loc='lower right', fontsize=9)

# ── Panel 2: Monthly delay trend ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
x = list(range(12))
vals = monthly_avg.values
ax2.plot(x, vals, color=C['blue'], linewidth=2.5, marker='D', markersize=8, zorder=5)
ax2.fill_between(x, vals, np.nanmin(vals)-1, alpha=0.12, color=C['blue'])
for xi, v in enumerate(vals):
    if not np.isnan(v):
        ax2.text(xi, v+0.3, f'{v:.1f}', ha='center', va='bottom',
                 fontsize=8.5, fontweight='bold')
pk = int(peak_month_i - 1)
ax2.scatter(pk, vals[pk], color=C['red'], s=160, zorder=6,
            label=f'Peak month: {peak_month_nm}')
ax2.set_xticks(x)
ax2.set_xticklabels(month_names, fontsize=10)
ax2.set_ylabel('Avg Delay (mins)', fontsize=11)
ax2.set_title('Panel 2 — Monthly Average Delay Trend', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=10)

# ── Panel 3a: Vehicle avg delay ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
veh = vehicle_perf.sort_values('avg_delay')
vcols = [C['blue'], C['purple'], C['green'], C['amber']][:len(veh)]
b3 = ax3.bar(veh.index, veh['avg_delay'], color=vcols, width=0.5,
             edgecolor='white', linewidth=1.5)
for b, v in zip(b3, veh['avg_delay']):
    ax3.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylabel('Avg Delay (mins)', fontsize=11)
ax3.set_title('Panel 3a — Vehicle: Avg Delay', fontsize=12, fontweight='bold', pad=8)

# ── Panel 3b: Vehicle on-time rate ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
b4 = ax4.bar(veh.index, veh['on_time_pct'], color=vcols, width=0.5,
             edgecolor='white', linewidth=1.5)
for b, v in zip(b4, veh['on_time_pct']):
    ax4.text(b.get_x()+b.get_width()/2, v+0.4, f'{v:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_ylabel('On-Time Rate (%)', fontsize=11)
ax4.set_title('Panel 3b — Vehicle: On-Time Rate', fontsize=12, fontweight='bold', pad=8)

plt.savefig('/home/claude/q4.png', dpi=150, bbox_inches='tight', facecolor=C['bg'])
plt.close()
print("  [✓] Q4 saved")

print(f"\n  ★ BIGGEST FIX: {worst_city} has the lowest on-time rate ({city_ontime.min():.1f}%).")
print(f"     {worst_vehicle}s have the highest avg delay. Surge in {peak_month_nm}.")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(14, 8))
fig2.patch.set_facecolor('#0F172A')
ax.set_facecolor('#0F172A'); ax.axis('off')

ax.text(0.5, 0.95, 'MINIHACK — KEY FINDINGS  (Real Dataset)',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=17, fontweight='bold', color='white')
ax.text(0.5, 0.88, 'Last Mile Delivery  |  2,080 rows  |  10 Cities  |  Full Year 2024',
        transform=ax.transAxes, ha='center', fontsize=10, color='#94A3B8')

kpis = [
    ("Q1 — Peak Delay Gap",    f"+{diff:.1f} mins",
     f"{pct_diff:.0f}% above off-peak  |  p={p:.0e}", C['red']),
    ("Q2 — Worst Weather",     "Rain",
     f"Median {w_med.get('Rain',0):.1f} min  |  {hardest} hardest hit", C['blue']),
    ("Q3 — Experience Gap",    f"+{g1.mean()-g2.mean():.1f} mins",
     f"p={p:.1e}  |  Cohen's d={d:.2f} ({size})", C['purple']),
    ("Q4 — City Fix Needed",   worst_city,
     f"{city_ontime.min():.1f}% on-time  |  Worst performer", C['amber']),
]
for i, (title, metric, detail, color) in enumerate(kpis):
    x0 = 0.03 + (i%2)*0.50
    y0 = 0.55 if i<2 else 0.20
    rect = FancyBboxPatch((x0,y0),0.44,0.27,
                          boxstyle="round,pad=0.01",
                          fc='#1E293B', ec=color, linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x0+0.22, y0+0.20, title, transform=ax.transAxes,
            ha='center', fontsize=10, color='#94A3B8', fontweight='bold')
    ax.text(x0+0.22, y0+0.13, metric, transform=ax.transAxes,
            ha='center', fontsize=20, color=color, fontweight='bold')
    ax.text(x0+0.22, y0+0.05, detail, transform=ax.transAxes,
            ha='center', fontsize=9, color='#CBD5E1')

ax.text(0.5, 0.10,
        f'★  Operational Fix: Deploy experienced riders (>4 yrs) during Peak Hours & Rain in {worst_city}',
        transform=ax.transAxes, ha='center', fontsize=10,
        color='#FDE68A', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='#1E293B', ec=C['amber']))

plt.savefig('/home/claude/summary.png', dpi=150, bbox_inches='tight', facecolor='#0F172A')
plt.close()
print("  [✓] Summary saved")
print("\n  ALL DONE ✓")