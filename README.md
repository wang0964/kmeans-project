# diamond_cluster_analysis

This app has been built with **Streamlit** and is deployed on **Streamlit Community Cloud**.

[Visit the app here](https://assignment-kmeans.streamlit.app//)  
*(No password required)*

The application groups diamonds into meaningful clusters based on their physical and quality attributes. It offers a clean interface where users can enter (or upload) diamond parameters and instantly see which cluster the gem belongs to, thanks to an unsupervised **K-Means** model.

---

## Features

- Intuitive Streamlit form for entering diamond attributes (carat, cut, color, clarity, proportions, dimensions)
- Real-time clustering result displayed as soon as the user clicks **Start Analysis**
- Consistent scaling with a pre-fitted `StandardScaler` to keep inputs in sync with the training data
- Deployed on Streamlit Cloud—no local installation needed to try it

---

## Dataset

The model was trained on the well-known **Diamonds Prices** dataset (originally from Kaggle). Key variables include:

- **carat** – weight of the diamond  
- **cut** – quality of the cut (Ideal, Premium, …)  
- **color** – color grading (D–J)  
- **clarity** – clarity scale (IF, VVS1, …)  
- **depth**, **table** – percentage proportions  
- **x, y, z** – physical dimensions in millimetres  

---

## Technologies Used

- **Streamlit** – interactive web UI
- **Scikit-learn** – data scaling (`StandardScaler`) and clustering (`KMeans`)
- **Pandas / NumPy** – data handling
- **Matplotlib / Seaborn** – optional exploratory charts

---

## Model

The application relies on a trained **K-Means** model with the following pipeline:

1. Map categorical features (`cut`, `color`, `clarity`) to integer codes  
2. Select numeric feature set `[carat, cut, color, clarity, depth, table, x, y, z]`  
3. Apply a `StandardScaler` (fit on training data)  
4. Fit **K-Means** with **3 clusters** (`n_init=20`, `random_state=42`)  

Both the fitted scaler (`scaler.pkl`) and the K-Means model (`kmeans_model.pkl`) are stored in `/models` and loaded at runtime for consistent predictions.

---

## Installation (local run)

```bash
# 1. Clone the repository
git clone https://github.com/wang0964/kmeans-project.git
cd kmeans-project

# 2. (Optional) Create & activate virtual environment
python -m venv env
# Windows:
env\Scripts\activate
# macOS / Linux:
source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run streamlit-app.py
```

---

## Dependencies

```txt
streamlit==1.46.0
pandas==2.3.0
scikit-learn==1.7.0
matplotlib==3.8.0
seaborn==0.12.2
```

---

#### Thank you for using the Diamond Cluster Analysis app!  
This project is submitted for **CST2216 Individual Term Project**.
