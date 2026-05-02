# quant_insight-ml-stock-decision-system
ML based stock decision support system with feature engineering, predictive modeling and real time Streamlit deployment

## features
- End to end ML pipeline (data->features-> model->prediction)
- Technical indicators: RSI, MACD, Moving Averages
- Predictive modeling using Random Forest
- Decision engine BUY/ SELL/ HOLD logic
- Risk analysis using volitility metrics
- Intereactive Streamlit dashboard

## Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- yfinance

## How it works
1. Fetch stock data using API  
2. Preprocess and clean data  
3. Generate financial features  
4. Train ML model  
5. Predict future price  
6. Generate decision (BUY / SELL / HOLD)

## 📊 Output
- Predicted price
- Expected return %
- Confidence score
- Risk level
- Decision recommendation

## Run Locally

```bash
git clone https://github.com/AmbujAg/quant_insight-ml-stock-decision-system.git
cd quant_insight-ml-stock-decision-system
pip install -r requirements.txt
streamlit run app.py
