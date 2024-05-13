import pickle
import matplotlib.pyplot as plt

with open('metrics.pkl', 'rb') as f:
    data = pickle.load(f)

losses = [entry['loss'] for entry in data]
iterations = range(0, len(losses) * 100, 100)  # Assuming the first iteration is at 0

plt.figure(figsize=(12, 6))
plt.plot(iterations, losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.show()
