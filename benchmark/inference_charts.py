import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados do CSV
data = pd.read_csv('inference_results.csv')
FONT_SIZE_1=16
FONT_SIZE_2=14

# Converter unidades de memória para valores numéricos (ex: 8g -> 8 * 1024 MB)
def convert_memory(mem):
    if 'g' in mem.lower():
        return int(mem.lower().replace('g', '')) * 1024
    elif 'm' in mem.lower():
        return int(mem.lower().replace('m', ''))
    return int(mem)

data['Memory_MB'] = data['Memory'].apply(convert_memory)

# Ajustar o tamanho da fonte geral
sns.set_context("notebook", font_scale=1.2)

# Gráfico 1: Relação entre CPUs e Throughput
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='CPU', y='Throughput', hue='Memory_MB', marker="o")
# plt.title("Relationship between CPUs and Throughput for Different Memory Sizes", fontsize=14)
plt.xlabel("Number of CPUs", fontsize=FONT_SIZE_1)
plt.ylabel("Throughput (inferences per second)", fontsize=FONT_SIZE_1)
plt.legend(title='Memory (MB)', fontsize=FONT_SIZE_2)
plt.grid(True)
plt.tight_layout()
plt.savefig('cpu_vs_throughput.png')
plt.show()

# Gráfico 2: Relação entre CPUs e Tempo Médio de Inferência
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='CPU', y='AvgInferenceTime', hue='Memory_MB', marker="o")
# plt.title("Relationship between CPUs and Average Inference Time for Different Memory Sizes", fontsize=14)
plt.xlabel("Number of CPUs", fontsize=FONT_SIZE_1)
plt.ylabel("Average Inference Time (s)", fontsize=FONT_SIZE_1)
plt.legend(title='Memory (MB)', fontsize=FONT_SIZE_2)
plt.grid(True)
plt.tight_layout()
plt.savefig('cpu_vs_avg_inference_time.png')
plt.show()

# Gráfico 3: Relação entre Memória e Throughput para diferentes CPUs
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Memory_MB', y='Throughput', hue='CPU', marker="o")
# plt.title("Relationship between Memory and Throughput for Different CPU Counts", fontsize=14)
plt.xlabel("Allocated Memory (MB)", fontsize=FONT_SIZE_1)
plt.ylabel("Throughput (inferences per second)", fontsize=FONT_SIZE_1)
plt.legend(title='Number of CPUs', fontsize=FONT_SIZE_2)
plt.grid(True)
plt.tight_layout()
plt.savefig('memory_vs_throughput.png')
plt.show()
