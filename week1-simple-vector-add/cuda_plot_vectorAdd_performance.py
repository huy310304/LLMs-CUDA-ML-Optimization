import matplotlib.pyplot as plt

# Data
vector_sizes = [1024, 1024 * 1024, 1024 * 1024 * 10, 1024 * 1024 * 100]
gpu_times = [0.000062, 0.000034, 0.000063, 0.000063]  # GPU times after running vector_add.cu 
cpu_times = [0.000010, 0.002921, 0.029464, 0.294553]  # CPU times after running vector_add.cu 

# Plotting stuff
plt.figure(figsize=(10, 5))
plt.plot(vector_sizes, gpu_times, label='GPU', marker='o')
plt.plot(vector_sizes, cpu_times, label='CPU', marker='o')
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Vector Size')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="--")

for i, (size, time) in enumerate(zip(vector_sizes, gpu_times)):
    plt.text(size, time, f'{time:.6f}', ha='center', va='bottom')
for i, (size, time) in enumerate(zip(vector_sizes, cpu_times)):
    plt.text(size, time, f'{time:.6f}', ha='center', va='bottom')

# Save and show the plot
plt.savefig('cuda_execution_time_vs_vector_size.png')
plt.show()

