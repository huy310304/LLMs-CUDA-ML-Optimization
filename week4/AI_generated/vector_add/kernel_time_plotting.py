import matplotlib.pyplot as plt

llms = ['Bard', 'ChatGPT-3.5', 'Claude'] 
execution_time = [20.416, 20.417, 20.352]  # Corresponding execution times in seconds for a fixed threads per block

# Create a plot
plt.figure(figsize=(10, 6))
plt.bar(llms, execution_time, color=['blue', 'green', 'red'])

# Add titles and labels
plt.title('Kernel Execution Time for Different LLMs with 1024*1024 vector size with 256 threadsPerBlocks')
plt.xlabel('LLMs')
plt.ylabel('Execution Time (us)')

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
