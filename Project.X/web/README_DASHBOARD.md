# 🚀 AQMSS TradingView-Style Dashboard

## Overview

A professional **AI-powered Market Quality Scoring System** with a modern TradingView-inspired interface. This dashboard provides real-time market analysis, AI insights, and trading recommendations.

## ✨ Features

### 🎨 TradingView-Inspired Design
- **Dark Theme** with professional color scheme
- **Responsive Layout** that works on all devices
- **Modern UI Components** with smooth animations
- **Real-time Updates** with auto-refresh capability

### 📊 Market Analysis
- **Quality Scoring** - Comprehensive market quality metrics (0-100)
- **Real-time Data** - Live market prices, volume, and conditions
- **Advanced Filtering** - Search and filter by quality, momentum, volatility
- **Sortable Tables** - Sort by any metric instantly
- **Detailed Views** - Click any market for in-depth analysis

### 🤖 AI Assistant
- **Machine Learning** powered predictions
- **Natural Language** chat interface  
- **Quick Actions** for common queries
- **Market Insights** generated automatically
- **Personalized Recommendations**

### 📈 Analytics & Charts
- **Quality Distribution** - Doughnut chart of market quality levels
- **Momentum Overview** - Bar chart of bullish/bearish markets
- **Volatility Analysis** - Pie chart of volatility distribution
- **Liquidity Trends** - Line chart of market liquidity

### 💡 AI Insights
- **Opportunity Detection** - Identifies high-quality trading opportunities
- **Risk Warnings** - Alerts for high volatility or poor conditions
- **Trend Analysis** - Bullish/bearish market signals
- **Priority System** - High/Medium/Low priority categorization

## 🏗️ Architecture

```
web/
├── app.py                 # Flask backend API
├── templates/
│   └── index.html        # Main dashboard interface
└── static/
    ├── styles.css        # TradingView-style CSS
    └── app.js            # Enhanced JavaScript logic
```

## 🔧 Technology Stack

**Frontend:**
- HTML5 with modern semantic structure
- CSS3 with custom properties & animations
- Vanilla JavaScript (ES6+)
- Chart.js for data visualization
- Axios for API requests

**Backend:**
- Python 3.8+
- Flask 2.3.0 - Web framework
- Flask-CORS - Cross-origin support
- Pandas - Data manipulation
- NumPy - Numerical computations

**AI/ML:**
- Scikit-learn - Machine learning models
- Custom quality scoring algorithms
- Real-time prediction engine

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web
python -m pip install -r requirements.txt
```

### 2. Launch Dashboard

```bash
python app.py
```

### 3. Open Browser

Navigate to: **http://localhost:5000**

## 📖 User Guide

### Main Features

#### 1. **Markets Tab** 📈
- View all markets with real-time data
- Search by symbol or name
- Filter by quality level (High/Medium/Low)
- Sort by any column (Price, Volume, Quality, etc.)
- Click "View" button for detailed analysis
- Enable Auto-refresh for live updates

#### 2. **Insights Tab** 💡
- AI-generated market opportunities
- High-priority alerts and warnings
- Bullish/bearish market signals
- Automatically updated

#### 3. **Analytics Tab** 📊
- Quality distribution visualization
- Momentum overview charts
- Volatility analysis
- Liquidity trends
- Interactive Chart.js visualizations

#### 4. **AI Assistant Tab** 🤖
- Chat with AI about markets
- Ask questions like:
  - "What are the best quality markets?"
  - "Show me bullish opportunities"
  - "How do I manage risk?"
  - "Analyze SYMBOL for me"
- Use Quick Action buttons
- Get AI predictions for any symbol
- Clear chat history

### Sidebar Widgets

**Market Overview:**
- Total Markets count
- Average Quality score
- High Quality markets count
- Bullish/Bearish breakdown
- Volatile markets alert

**AI Market Pulse:**
- Real-time AI summary
- Overall market sentiment
- Key alerts and signals

## 🎯 API Endpoints

### GET `/api/markets`
Get all markets with quality scores
```json
{
  "success": true,
  "data": [...],
  "count": 10,
  "timestamp": "2026-02-16T..."
}
```

### GET `/api/market/<symbol>`
Get detailed data for specific market

### POST `/api/ai/chat`
Chat with AI assistant
```json
{
  "message": "What are the best markets?"
}
```

### POST `/api/ai/predict`
Get AI prediction for market
```json
{
  "symbol": "BTCUSD",
  "volatility_regime": "NORMAL",
  "liquidity": 75.5,
  ...
}
```

### GET `/api/ai/insights`
Get AI-generated insights

### GET `/api/dashboard/stats`
Get dashboard statistics

## 🎨 Design Features

### Color Scheme (TradingView-Inspired)
- **Background**: Dark theme (#0D1117, #161B22)
- **Primary Blue**: #2962FF (TradingView signature blue)
- **Success Green**: #26A69A (Bullish)
- **Danger Red**: #EF5350 (Bearish)
- **Warning Orange**: #FF9800
- **Text**: Light (#E6EDF3) with hierarchy

### Typography
- **Font**: Inter (Professional, modern)
- **Weights**: 300-700 for hierarchy
- **Sizes**: Responsive with rem units

### Components
- **Cards**: Elevated with shadows
- **Buttons**: Hover effects & transitions
- **Tables**: Striped, sortable, hoverable
- **Charts**: Responsive Chart.js visualizations
- **Modal**: Overlay for detailed views

## 🔐 Security

- Input sanitization (XSS protection)
- CORS configuration
- Environment-based configuration
- No sensitive data exposure

## 📊 Quality Score Breakdown

The AI calculates quality scores (0-100) based on:

1. **Liquidity** (30%) - Order book depth, volume
2. **Volatility** (25%) - ATR, price stability
3. **Momentum** (25%) - Trend strength, direction
4. **Volume** (20%) - Trading activity, participation

**Score Ranges:**
- **75-100**: Excellent (Green) - Optimal conditions
- **60-74**: Good (Light Green) - Favorable
- **45-59**: Medium (Orange) - Caution advised
- **0-44**: Poor (Red) - High risk

## 🤖 AI Assistant Capabilities

The AI can help with:
- Market quality analysis
- Trading strategy recommendations
- Risk management advice
- Technical indicator explanations
- Volatility assessment
- Liquidity analysis
- Trend identification
- Opportunity discovery

**Sample Questions:**
```
"What's the quality score?"
"Show me the best markets"
"Which markets are most volatile?"
"Analyze BTCUSD"
"How do I manage risk?"
"What's your trading strategy?"
```

## 🔄 Auto-Refresh

Enable auto-refresh to update data every 30 seconds:
1. Click the **Auto** button in the Markets tab
2. Button turns green when active
3. All data refreshes automatically
4. Click again to disable

## 📱 Responsive Design

Fully responsive for:
- **Desktop** (1920px+): Full layout with sidebar
- **Tablet** (768px-1200px): Stacked layout
- **Mobile** (<768px): Single column, optimized controls

## 🎓 Best Practices

1. **Use quality scores > 70** for optimal trading
2. **Monitor volatility** before entering positions
3. **Check liquidity** for large orders
4. **Follow AI recommendations** as guidance, not rules
5. **Enable auto-refresh** for live monitoring
6. **Ask the AI** when uncertain

## 🐛 Troubleshooting

**Markets not loading:**
- Check if `results/market_scores.csv` exists
- Verify Flask server is running
- Check browser console for errors

**Charts not displaying:**
- Ensure you're on the Analytics tab
- Verify Chart.js is loaded
- Check data is available

**AI not responding:**
- Check Flask API is running
- Verify network tab for API calls
- Check chat input is not empty

## 🔮 Future Enhancements

- [ ] Real-time WebSocket updates
- [ ] User authentication & portfolios
- [ ] Customizable watchlists
- [ ] Advanced charting (candlesticks, indicators)
- [ ] Email/SMS alerts
- [ ] Historical data analysis
- [ ] Backtesting capabilities
- [ ] Multi-exchange support
- [ ] Mobile app (React Native)
- [ ] API rate limiting

## 📝 License

Proprietary - AQMSS Project

## 👨‍💻 Developer

Built with ❤️ using AI assistance

---

**🚀 Ready to analyze markets professionally!**

Open http://localhost:5000 and start exploring your AI-powered trading dashboard.
