import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

df = pd.read_csv("/Users/srujanakasipuram/Downloads/PS_20174392719_1491204439457_log.csv")

# Keep only meaningful transaction types
df = df[df["type"].isin(["PAYMENT", "CASH_OUT"])]

# Encode transaction type
df["type"] = df["type"].map({"PAYMENT": 0, "CASH_OUT": 1})

# Select relevant columns
df = df[
    [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud"
    ]
]

# Sample for feasibility
df = df.sample(200000, random_state=42).reset_index(drop=True)

df.to_csv("/Users/srujanakasipuram/Documents/Project/clean_data.csv", index=False)
print("Clean data saved")

df["balance_gap"] = (
    df["oldbalanceOrg"]
    - df["newbalanceOrig"]
    - df["amount"]
).abs()

df["amount_log"] = np.log1p(df["amount"])

df["dest_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

split_step = df["step"].quantile(0.8)

train = df[df["step"] <= df["step"].quantile(0.8)]
test = df[df["step"] > df["step"].quantile(0.8)]

train.to_csv("/Users/srujanakasipuram/Documents/Project/train_data.csv", index=False)
test.to_csv("/Users/srujanakasipuram/Documents/Project/test_data.csv", index=False)

print("Train size:", train.shape)
print("Test size :", test.shape)


def build_graph(df):
    accounts = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
    tx_ids = df.index.astype(str)

    acc_nodes = [f"acc_{a}" for a in accounts]
    tx_nodes = [f"tx_{i}" for i in tx_ids]

    all_nodes = acc_nodes + tx_nodes
    node_map = {n: i for i, n in enumerate(all_nodes)}

    x = []

    # Account nodes (structural only)
    for _ in acc_nodes:
        x.append([0, 0, 0, 0, 0])

    # Transaction nodes (ADVANCED FEATURES)
    for _, r in df.iterrows():
        x.append([
            r["amount_log"],
            r["type"],
            r["balance_gap"],
            r["dest_balance_change"],
            r["step"]
        ])

    x = torch.tensor(x, dtype=torch.float)

    y = torch.zeros(len(all_nodes))
    for i, r in df.iterrows():
        y[node_map[f"tx_{i}"]] = r["isFraud"]

    edges = []
    for i, r in df.iterrows():
        edges.append([node_map[f"acc_{r['nameOrig']}"], node_map[f"tx_{i}"]])
        edges.append([node_map[f"tx_{i}"], node_map[f"acc_{r['nameDest']}"]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, y=y)

train_graph = build_graph(train)
test_graph  = build_graph(test)

torch.save(train_graph, "/Users/srujanakasipuram/Documents/Project/train_graph.pt")
torch.save(test_graph, "/Users/srujanakasipuram/Documents/Project/test_graph.pt")

graph = torch.load(
    "/Users/srujanakasipuram/Documents/Project/train_graph.pt",
    weights_only=False
)

tx_mask = graph.x[:, 0] > 0
tx_idx = tx_mask.nonzero(as_tuple=True)[0]

perm = torch.randperm(len(tx_idx))
split = int(0.8 * len(tx_idx))

graph.train_mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
graph.val_mask   = torch.zeros(graph.num_nodes, dtype=torch.bool)

graph.train_mask[tx_idx[perm[:split]]] = True
graph.val_mask[tx_idx[perm[split:]]] = True

class AdvancedFraudGNN(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)

        self.conv2 = SAGEConv(128, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)

        self.conv3 = SAGEConv(128, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)

        self.out = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index):
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, 0.4, self.training)

        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h2 = h2 + h1   # residual
        h2 = F.dropout(h2, 0.4, self.training)

        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))

        return self.out(h3).squeeze()
    
num_fraud = graph.y[graph.train_mask].sum()
num_legit = graph.train_mask.sum() - num_fraud

pos_weight = num_legit / num_fraud

model = AdvancedFraudGNN(graph.x.size(1))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

loss_fn = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

best_auc = 0
MODEL_PATH = "/Users/srujanakasipuram/Documents/Project/best_model.pt"

for epoch in range(40):
    model.train()
    optimizer.zero_grad()

    logits = model(graph.x, graph.edge_index)

    loss = loss_fn(
        logits[graph.train_mask],
        graph.y[graph.train_mask]
    )

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(logits[graph.val_mask])
        auc = roc_auc_score(
            graph.y[graph.val_mask].cpu(),
            probs.cpu()
        )

    print(f"Epoch {epoch:02d} | Loss {loss:.4f} | AUC {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_PATH)

model.load_state_dict(
    torch.load(MODEL_PATH, weights_only=True)
)
model.eval()

test_graph = torch.load(
    "/Users/srujanakasipuram/Documents/Project/test_graph.pt",
    weights_only=False
)

with torch.no_grad():
    logits = model(test_graph.x, test_graph.edge_index)
    probs = torch.sigmoid(logits)

mask = test_graph.x[:,0] > 0

print("TEST ROC-AUC:", roc_auc_score(test_graph.y[mask], probs[mask]))
print("TEST PR-AUC :", average_precision_score(test_graph.y[mask], probs[mask]))
print(classification_report(test_graph.y[mask], probs[mask] > 0.7))

def predict_transaction(node_id):
    if node_id < 0 or node_id >= graph.num_nodes:
        return {"error": "Invalid node_id"}

    if graph.x[node_id][0].item() == 0:
        return {"error": "Node is not a transaction node"}

    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        prob = torch.sigmoid(logits[node_id]).item()

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= 0.7
    }

tx_nodes = (graph.x[:,0] > 0).nonzero(as_tuple=True)[0]
print("Sample transaction node_ids:", tx_nodes[:20].tolist())

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# -------------------------------
# HEALTH CHECK
# -------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "Fraud Detection API",
        "time": datetime.utcnow().isoformat()
    })

# -------------------------------
# SINGLE PREDICTION
# -------------------------------
@app.route("/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        if not data or "node_id" not in data:
            return jsonify({
                "error": "node_id is required"
            }), 400

        node_id = int(data["node_id"])

        result = predict_transaction(node_id)

        if "error" in result:
            return jsonify(result), 400

        return jsonify({
            "node_id": node_id,
            "prediction": result["is_fraud"],
            "fraud_probability": result["fraud_probability"],
            "risk_level": (
                "HIGH" if result["fraud_probability"] > 0.8 else
                "MEDIUM" if result["fraud_probability"] > 0.5 else
                "LOW"
            ),
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

# -------------------------------
# PREDICTION + EXPLANATION
# -------------------------------
@app.route("/predict/explain", methods=["POST"])
def api_predict_explain():
    try:
        data = request.get_json()
        node_id = int(data.get("node_id", -1))

        if node_id < 0:
            return jsonify({"error": "Invalid node_id"}), 400

        prediction = predict_transaction(node_id)
        explanation = explain_transaction(graph, node_id)

        return jsonify({
            "node_id": node_id,
            "fraud_probability": prediction["fraud_probability"],
            "is_fraud": prediction["is_fraud"],
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# -------------------------------
# BATCH PREDICTION
# -------------------------------
@app.route("/predict/batch", methods=["POST"])
def api_predict_batch():
    try:
        data = request.get_json()
        node_ids = data.get("node_ids", [])

        if not isinstance(node_ids, list):
            return jsonify({"error": "node_ids must be a list"}), 400

        results = []
        for nid in node_ids:
            pred = predict_transaction(int(nid))
            results.append({
                "node_id": nid,
                "fraud_probability": pred["fraud_probability"],
                "is_fraud": pred["is_fraud"]
            })

        return jsonify({
            "count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# MODEL INFO
# -------------------------------
@app.route("/model/info", methods=["GET"])
def model_info():
    return jsonify({
        "model": "Graph Neural Network (GraphSAGE)",
        "task": "Online Fraud Detection",
        "node_type": "Transaction Nodes",
        "output": "Fraud Probability",
        "threshold": 0.7
    })

# -------------------------------
# SERVER START
# -------------------------------
if __name__ == "__main__":
    print("Advanced Fraud Detection API running...")
    app.run(host="0.0.0.0", port=6006, debug=True)