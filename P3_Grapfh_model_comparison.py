import matplotlib.pyplot as plt
import json
import glob

colors = ['red', 'blue', 'green', 'darkorange', 'purple', 'brown']

plt.figure(figsize=(10, 6))

json_files = glob.glob("loss_*.json") #JSON files in directory starting with loss_
json_files.sort()

if not json_files:
    print("No JSON-files found in directory")
else:
    for i, file_path in enumerate(json_files):
        with open(file_path, "r") as f:
            loss_data = json.load(f)
            label_name = file_path.replace("loss_", "").replace(".json", "").replace("_", ", ")
            current_color = colors[i % len(colors)]
            plt.plot(loss_data, color=current_color, label=f"Layers: ({label_name})", linewidth=2)

# setting up graph
plt.title("MLP Model Comparison: Loss Curves")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.grid(True)
plt.legend()

# saving graph
plt.savefig("mlp_model_comparison.png")
print(f"Graph saved based on {len(json_files)} file(s).")

plt.show()