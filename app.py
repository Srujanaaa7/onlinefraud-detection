from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# GNN MODEL DEFINITION
# -------------------------------------------------
class FraudGNN(torch.nn.Module):
    def __init__(self):
        super(FraudGNN, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.softmax(x, dim=1)

# Load model
model = FraudGNN()
model.eval()

# -------------------------------------------------
# FRONTEND (HTML)
# -------------------------------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Online Fraud Detection Using GNN</title>
<style>
body { font-family: Arial; background:#f4f6f9; }
.header { background:#2c3e50; color:white; padding:20px; text-align:center; }
.container { width:90%; margin:30px auto; }
.card { background:white; padding:25px; border-radius:6px; box-shadow:0 4px 8px rgba(0,0,0,.1); }
input,select { width:100%; padding:10px; margin:8px 0; }
button { padding:10px 25px; background:#3498db; color:white; border:none; cursor:pointer; }
.safe { color:green; font-weight:bold; }
.fraud { color:red; font-weight:bold; }
</style>
</head>

<body>
<div class="header">
<h1>Online Fraud Detection Using GNN</h1>
<p>Graph Neural Network Based Transaction Analysis</p>
</div>

<div class="container">
<div class="card">
<h2>Transaction Details</h2>
<input type="number" id="amount" placeholder="Transaction Amount">
<input type="number" id="connections" placeholder="Connected Accounts">
<button onclick="checkFraud()">Check Fraud</button>
<div id="output"></div>
</div>
</div>

<script>
function checkFraud(){
    let amount = document.getElementById("amount").value;
    let connections = document.getElementById("connections").value;

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({
            amount: amount,
            connections: connections
        })
    })
    .then(res => res.json())
    .then(data => {
        let out = document.getElementById("output");
        if(data.prediction === "fraud"){
            out.innerHTML = "⚠ Fraud Detected<br>Confidence: " + data.score;
            out.className = "fraud";
        } else {
            out.innerHTML = "✔ Legitimate Transaction<br>Confidence: " + data.score;
            out.className = "safe";
        }
    });
}
</script>
</body>
</html>
"""

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    amount = float(data["amount"])
    connections = float(data["connections"])

    # Node feature: [amount, connections]
    x = torch.tensor([[amount, connections]], dtype=torch.float)

    # Simple self-loop graph
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # GNN forward pass
    output = model(x, edge_index)
    prob = output[0].detach().numpy()

    if prob[1] > prob[0]:
        prediction = "fraud"
        score = round(float(prob[1]), 3)
    else:
        prediction = "legitimate"
        score = round(float(prob[0]), 3)

    return jsonify({
        "prediction": prediction,
        "score": score
    })

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)