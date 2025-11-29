# ---------------------------------------------------------
# PREMIUM REAL ESTATE INVESTMENT ADVISOR – FULL APP.PY
# Premium UI • SHAP • Insights • Fallback models
# ---------------------------------------------------------

import streamlit as st
import joblib, os
import pandas as pd, numpy as np
import plotly.express as px
import requests
import shap
import scipy.sparse as sp
from streamlit.components.v1 import html

# ---------------------------------------------------------
# FALLBACK MODELS
# ---------------------------------------------------------

class RuleClassifier:
    def __init__(self, median_pps):
        self.median_pps = median_pps

    def predict(self, X):
        pps = pd.to_numeric(X.get("Price_per_SqFt"), errors="coerce").fillna(self.median_pps)
        bhk = pd.to_numeric(X.get("BHK"), errors="coerce").fillna(3)
        preds = ((pps <= self.median_pps) & (bhk >= 3)).astype(int)
        return preds.values

    def predict_proba(self, X):
        preds = self.predict(X)
        return np.vstack([1 - preds, preds]).T


class RuleRegressor:
    def __init__(self, growth_rate=0.08):
        self.growth_rate = growth_rate

    def predict(self, X):
        price = pd.to_numeric(X.get("Price_in_Lakhs"), errors="coerce").fillna(0)
        return price * ((1 + self.growth_rate) ** 5)


# ---------------------------------------------------------
# STREAMLIT PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor (Premium)",
    page_icon="🏙️",
    layout="wide"
)

# ---------------------------------------------------------
# CSS FOR BEAUTIFUL UI
# ---------------------------------------------------------
st.markdown("""
<style>
.app-card {
    background: rgba(255,255,255,0.85);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}
.metric-card {
    background: rgba(255,255,255,0.9);
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stButton button {
    background-color:#0b3d91;
    color:white;
    padding:10px 25px;
    border-radius:8px;
    border:none;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}
.stButton button:hover {
    background-color:#072b6e;
    transform:scale(1.05);
}
.small-muted { color:#666; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOTTIE HEADER
# ---------------------------------------------------------
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.text
    except:
        return None

lottie_json = load_lottie("https://assets2.lottiefiles.com/packages/lf20_touohxv0.json")

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("<h1 style='color:#0b3d91;'>🏙 Real Estate Investment Advisor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Premium dashboard • Investment classification + 5-year price forecast</div>", unsafe_allow_html=True)

with col2:
    if lottie_json:
        html(f"""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.7.6/lottie.min.js"></script>
        <div id="lottie" style="width:140px;height:140px;"></div>
        <script>
        var anim = {lottie_json};
        lottie.loadAnimation({{
            container: document.getElementById('lottie'),
            animationData: anim,
            renderer: 'svg',
            loop:true,
            autoplay:true
        }});
        </script>
        """, height=150)

# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------
def app_root():
    return os.path.dirname(os.path.abspath(__file__))

def paths():
    base = app_root()
    processed = os.path.join(base, "..", "data", "india_housing_prices_processed.csv")
    raw = os.path.join(base, "..", "data", "india_housing_prices.csv")
    return processed, raw

@st.cache_data
def load_data():
    processed, raw = paths()
    if os.path.exists(processed):
        return pd.read_csv(processed)
    if os.path.exists(raw):
        return pd.read_csv(raw)
    st.error("❌ Dataset missing!")
    st.stop()

def load_model(name):
    path = os.path.join(app_root(), "..", "models", name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ---------------------------------------------------------
# LOAD DATA + MODELS
# ---------------------------------------------------------
df = load_data()

if "Price_per_SqFt" not in df.columns:
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]

clf = load_model("xgb_classifier.pkl")
reg = load_model("xgb_regressor.pkl")

if clf is None:
    clf = RuleClassifier(df["Price_per_SqFt"].median())
if reg is None:
    reg = RuleRegressor(0.08)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
home, insights, explain = st.tabs(["🏠 Home", "📊 Insights", "🧠 Explainability (SHAP)"])

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔎 Property Details")

    city = st.selectbox("City", sorted(df["City"].unique()))
    locality = st.selectbox("Locality", sorted(df["Locality"].unique()))
    ptype = st.selectbox("Property Type", sorted(df["Property_Type"].unique()))
    bhk = st.slider("BHK", 1, 6, 3)
    size = st.number_input("Size (SqFt)", value=900)
    price_lakhs = st.number_input("Current Price (Lakhs)", value=50.0)
    furnished = st.selectbox("Furnished", sorted(df["Furnished_Status"].unique()))
    nearby_schools = st.slider("Nearby Schools", 0, 20, 2)
    nearby_hospitals = st.slider("Nearby Hospitals", 0, 10, 1)
    transport = st.selectbox("Public Transport", sorted(df["Public_Transport_Accessibility"].unique()))
    parking = st.slider("Parking Spaces", 0, 5, 1)

    predict_btn = st.button("🔮 Analyze Property")

# ---------------------------------------------------------
# HOME TAB
# ---------------------------------------------------------
with home:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("Market Snapshot")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median Price (Lakhs)", f"{df['Price_in_Lakhs'].median():.2f}")
    c2.metric("Median Price/SqFt", f"{df['Price_per_SqFt'].median():.0f}")
    c3.metric("Avg Nearby Schools", f"{df['Nearby_Schools'].mean():.1f}")
    c4.metric("Available Listings", str((df["Availability_Status"] == "Available").sum()))

    st.markdown("</div><br>", unsafe_allow_html=True)

    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("Property Evaluation")

    if predict_btn:
        pps = price_lakhs * 100000 / max(size, 1)

        input_df = pd.DataFrame([{
            "City": city,
            "Locality": locality,
            "Property_Type": ptype,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price_lakhs,
            "Price_per_SqFt": pps,
            "Age_of_Property": 5,
            "Nearby_Schools": nearby_schools,
            "Nearby_Hospitals": nearby_hospitals,
            "Public_Transport_Accessibility": transport,
            "Parking_Space": parking,
            "Furnished_Status": furnished
        }])

        pred = int(clf.predict(input_df)[0])
        try:
            conf = clf.predict_proba(input_df)[0][1]
        except:
            conf = None

        price_5y = float(reg.predict(input_df)[0])

        def grade(conf):
            if conf is None: return "B"
            sc = conf * 100
            if sc > 80: return "AAA"
            if sc > 65: return "AA"
            if sc > 45: return "A"
            if sc > 30: return "B"
            return "C"

        rating = grade(conf)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Verdict")
            st.write("🏆 **Recommended**" if pred == 1 else "⚠️ **Not Recommended**")
            if conf: st.write(f"Confidence: {conf:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("5-Year Price")
            st.write(f"₹ {price_5y:.2f} Lakhs")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.subheader("Rating")
            st.write(f"**{rating}**")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# INSIGHTS TAB
# ---------------------------------------------------------
with insights:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("City Price Comparison")

    fig = px.bar(
        df.groupby("City")["Price_per_SqFt"].median().reset_index(),
        x="City",
        y="Price_per_SqFt",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Locality Bubble Chart")
    if "Locality" in df.columns and "ID" in df.columns:
        loc = df.groupby(["Locality", "City"]).agg(
            median_price=("Price_in_Lakhs", "median"),
            count=("ID", "count"),
            median_pps=("Price_per_SqFt", "median")
        ).reset_index()

        fig2 = px.scatter(
            loc,
            x="count",
            y="median_price",
            size="median_pps",
            color="City",
            hover_name="Locality",
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SHAP TAB (FIXED VERSION)
# ---------------------------------------------------------
with explain:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.subheader("SHAP Explainability")

    try:
        xgb_model = clf.named_steps["xgb"]
        preprocessor = clf.named_steps["pre"]
    except:
        st.error("Train XGBoost model first.")
        st.stop()

    st.info("SHAP plot generating...")

    sample = df.sample(150, random_state=42)

    features = [
        "City","Locality","Property_Type","BHK","Size_in_SqFt",
        "Price_in_Lakhs","Price_per_SqFt","Age_of_Property",
        "Nearby_Schools","Nearby_Hospitals",
        "Public_Transport_Accessibility","Parking_Space","Furnished_Status"
    ]

    available = [c for c in features if c in sample.columns]

    X_raw = sample[available].copy()

    # numeric
    nums = X_raw.select_dtypes(include=[np.number]).columns
    X_raw[nums] = X_raw[nums].fillna(X_raw[nums].median())

    # categorical
    cats = X_raw.select_dtypes(include=['object']).columns
    for c in cats:
        X_raw[c] = X_raw[c].fillna(df[c].mode()[0])

    Xt = preprocessor.transform(X_raw)

    if sp.issparse(Xt):
        Xt = Xt.toarray()

    explainer = shap.Explainer(xgb_model, Xt)
    shap_values = explainer(Xt)

    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, Xt, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown(
    "<br><center><div class='small-muted'>Built by Pruthviraj • Premium Real Estate AI Advisor</div></center>",
    unsafe_allow_html=True
)
