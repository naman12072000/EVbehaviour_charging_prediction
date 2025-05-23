import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Internshipprojects\datasets\Global_EV_Charging_Behavior_2024.csv")

# Fix column name typos (if needed)
df.columns = df.columns.str.replace('Charging', 'Charging')  # Correct any inconsistencies

# 1. Charging Station Type Distribution
plt.figure(figsize=(10, 5))
df['Charging Station Type'].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title("Number of Sessions per Charging Type", pad=20)
plt.xlabel("Charging Type")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# 2. Average Charging Duration by Type
plt.figure(figsize=(10, 5))
df.groupby('Charging Station Type')['Charging Duration (mins)'].mean().sort_values().plot(
    kind='bar', color='orange', edgecolor='black')
plt.title("Average Charging Time by Station Type", pad=20)
plt.xlabel("Charging Type")
plt.ylabel("Minutes")
plt.xticks(rotation=0)
plt.show()

# 3. Energy vs Duration (Colored by Type)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Charging Duration (mins)', y='Energy Delivered (kWh)', 
               hue='Charging Station Type', palette='viridis', alpha=0.7)
plt.title("Energy Delivered vs Charging Time", pad=20)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. Cost Comparison by Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Charging Station Type', y='Charging Cost ($)', palette='Set2')
plt.title("Charging Cost Distribution by Type", pad=20)
plt.xlabel("Charging Type")
plt.ylabel("Cost ($)")
plt.show()

# Temperature Analysis ======================================

# Create temperature bins first (for boxplot)
df['Temp_Bin'] = pd.cut(df['Temperature (°C)'], 
                       bins=[-10, 0, 10, 20, 30, 40],
                       labels=['<0°C', '0-10°C', '10-20°C', '20-30°C', '>30°C'])

# 5. Temperature vs Energy
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Temperature (°C)', y='Energy Delivered (kWh)', 
           scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Temperature Effect on Energy Delivered", pad=20)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Temperature (°C)', y='Energy Delivered (kWh)', 
           scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title("Temperature Effect on Energy Delivered")
plt.grid(True)
plt.show()

# 6. Charging Duration by Temperature Ranges
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Temp_Bin', y='Charging Duration (mins)', palette='coolwarm')
plt.title("Charging Duration Across Temperature Ranges", pad=20)
plt.xlabel("Temperature Range (°C)")
plt.ylabel("Duration (mins)")
plt.show()

# 7. Efficiency Analysis
df['Efficiency'] = df['Energy Delivered (kWh)'] / df['Battery Capacity (kWh)']
df['Efficiency'] = df['Efficiency'].clip(0, 1)  # Ensure realistic values

plt.figure(figsize=(10, 6))
sns.lineplot(data=df.groupby('Temperature (°C)')['Efficiency'].mean().reset_index(),
            x='Temperature (°C)', y='Efficiency',
            marker='o', color='green')
plt.title("Charging Efficiency vs Temperature", pad=20)
plt.ylabel("Efficiency (Energy/Battery)")
plt.grid(True)
plt.show()

# 8. Combined Temperature Effects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x='Temperature (°C)', y='Energy Delivered (kWh)', 
               ax=ax1, alpha=0.4, color='blue')
ax1.set_title("Energy vs Temperature")
ax1.set_ylabel("Energy (kWh)")

sns.scatterplot(data=df, x='Temperature (°C)', y='Charging Duration (mins)', 
               ax=ax2, alpha=0.4, color='red')
ax2.set_title("Duration vs Temperature")
ax2.set_ylabel("Duration (mins)")

plt.suptitle("Temperature Effects on Charging", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()