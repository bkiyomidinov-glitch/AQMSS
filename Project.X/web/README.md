# AQMSS Web Dashboard

A **TradingView-inspired AI Market Quality Scoring System** dashboard - modern, responsive web interface for real-time market analysis powered by AI.

![Dashboard Preview](#)

## 📊 Features

- **Real-time Market Data**: Live market rates, prices, and volumes displayed in an interactive table
- **TradingView-Style Design**: Professional, dark-themed interface inspired by TradingView
- **AI Quality Scoring**: Machine learning-based market quality assessment (0-100 score)
- **Market Analytics**: 
  - Sortable/filterable market table
  - Quality score distribution charts
  - Momentum analysis
  - Volatility tracking
- **AI Assistant Chat**: Interactive AI that provides trading insights and market analysis
- **Real-time Insights**: AI-generated market opportunities and warnings
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)

### Installation

1. **Navigate to web directory**
```bash
cd web
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Flask server**
```bash
python app.py
```

4. **Access dashboard**
Open your browser and navigate to:
```
http://localhost:5000
```

## 🐳 Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up -d
```

2. **Access the dashboard**
```
http://localhost:5000
```

3. **Stop the container**
```bash
docker-compose down
```

## 📁 Project Structure

```
web/
├── app.py                 # Flask backend API
├── templates/
│   └── index.html        # Main dashboard HTML
├── static/
│   ├── styles.css        # TradingView-inspired styling
│   └── app.js            # Frontend JavaScript/logic
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker image configuration
├── docker-compose.yml   # Docker Compose setup
└── README.md           # This file
```

## 📡 API Endpoints

### Markets
- `GET /api/markets` - Get all market data with scores
- `GET /api/market/<symbol>` - Get detailed market information

### Analytics
- `GET /api/dashboard/stats` - Dashboard statistics
- `GET /api/ai/insights` - AI-generated market insights

### AI Features
- `POST /api/ai/chat` - Chat with AI assistant
- `POST /api/ai/predict` - Get AI market quality prediction

## 🎯 Usage Guide

### Markets Tab
1. **View All Markets**: Sorted by symbol with real-time quality scores
2. **Search**: Filter markets by symbol or name
3. **Filter**: Quick filter by quality level (High/Medium/Low)
4. **Sort**: Click column headers to sort by price, change, volume, etc.
5. **Details**: Click "View" button to see detailed market information

### Insights Tab
- AI-generated market opportunities and warnings
- Priority levels (HIGH/MEDIUM) for quick action
- Categories: Bullish moves, High quality opportunities, Volatility alerts

### Analytics Tab
- **Quality Distribution**: Pie chart showing quality score distribution
- **Momentum Overview**: Bar chart of bullish vs bearish markets
- Real-time updates of market conditions

### AI Assistant Tab
- Ask questions about market quality and trends
- Get trading recommendations
- Learn about market conditions
- Example questions:
  - "What's the market quality score?"
  - "Which markets are bullish?"
  - "What about volatility?"
  - "Recommend a strategy"

## 🎨 Design Features

### Color Scheme
- **Primary**: `#1f77b4` - Main brand color (TradingView blue)
- **Bullish**: `#26a69a` - Green for positive trends
- **Bearish**: `#ef5350` - Red for negative trends
- **Warning**: `#ff7f0e` - Orange for high volatility
- **Background**: `#0a1929` - Dark theme for reduced eye strain

### Typography
- Clean, modern sans-serif font stack
- Optimal contrast for readability
- Responsive font sizes for all devices

### Responsive Breakpoints
- Desktop: Full layout with sidebar
- Tablet: Flexible grid layout
- Mobile: Single column, optimized for touch

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the `web/` directory:

```env
FLASK_ENV=production
FLASK_DEBUG=False
API_HOST=0.0.0.0
API_PORT=5000
```

### Data Source
The dashboard pulls data from:
- `../results/market_scores.csv` - Market quality scores
- `../models/metadata.json` - Model configuration
- `../data/dataset.csv` - Historical market data

## 📊 Data Format

Expected CSV structure (`market_scores.csv`):

```
symbol,name,price,change_percent,volume,market_quality_score,liquidity,volatility,momentum,market_condition,volatility_regime,trend,current_atr,recent_volume
```

## 🤖 AI Integration

The dashboard integrates with your existing AI modules:

- **ai_module.py**: Market quality prediction using Random Forest
- **predictor.py**: Market condition prediction
- **evaluator.py**: Market metrics evaluation

The AI assistant uses these models to:
1. Predict market quality probability
2. Generate trading insights
3. Provide market recommendations

## 🔐 Security Considerations

- CORS enabled for development (restrict in production)
- Input validation on all endpoints
- Escaped HTML to prevent XSS attacks
- Error handling for graceful failures

## 🚀 Performance Tips

- Use CDN for Chart.js and Axios libraries
- Implement data pagination for large datasets
- Cache market data for 5-10 second intervals
- Use WebSockets for real-time (future enhancement)

## 🐛 Troubleshooting

### Dashboard not loading?
- Check Flask server is running: `http://localhost:5000`
- Check browser console for errors (F12)
- Verify CORS settings in Flask app

### Market data empty?
- Ensure `../results/market_scores.csv` exists
- Check file permissions
- Verify data format matches expected schema

### AI features not working?
- Check AI model files exist in `../models/`
- Verify required Python packages installed
- Check `ai_module.py` has correct feature columns

## 📈 Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] User authentication and portfolios
- [ ] Advanced charting (TradingView Lightweight Charts)
- [ ] Alerts and notifications
- [ ] Historical data export
- [ ] Custom indicators
- [ ] Multi-timeframe analysis
- [ ] Machine learning model updates

## 📝 License

Part of Project AQMSS - AI Market Quality Scoring System

## 🤝 Support

For issues or questions:
1. Check existing issues in project repo
2. Review logs in browser console
3. Check Flask server output
4. Verify data files exist and are readable

## 📚 Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [TradingView Design System](https://www.tradingview.com/)
- [Chart.js Documentation](https://www.chartjs.org/)
- [Axios Documentation](https://axios-http.com/)

---

**AQMSS Dashboard v1.0** - Built with ❤️ for traders and analysts
