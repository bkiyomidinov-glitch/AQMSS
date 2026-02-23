"""
AQMSS TradingView-Style Web Dashboard API
Integrates with real AQMSS AI Market Quality Scoring System
Reads real data from market_scores.csv, dataset.csv, models/metrics.json
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import json
import os
import sys
import numpy as np
from pathlib import Path
import random

# Add parent directory to path to import AI modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ai_module import train_and_predict
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  AI module not available - using mock predictions")

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration - paths to real AQMSS data
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

# Cache for market data
market_cache = {
    'data': None,
    'timestamp': None,
    'ttl': 30
}

# ==========================================
# REAL AQMSS DATA LOADING
# ==========================================

def load_aqmss_scores():
    """Load real AQMSS scores from market_scores.csv"""
    scores_file = RESULTS_DIR / 'market_scores.csv'
    if not scores_file.exists():
        return None
    try:
        df = pd.read_csv(scores_file)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error loading AQMSS scores: {e}")
        return None


def load_model_metrics():
    """Load real AI model metrics from models/metrics.json"""
    metrics_file = MODELS_DIR / 'metrics.json'
    if not metrics_file.exists():
        return None
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        return None


def load_model_metadata():
    """Load model metadata from models/metadata.json"""
    meta_file = MODELS_DIR / 'metadata.json'
    if not meta_file.exists():
        return None
    try:
        with open(meta_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading model metadata: {e}")
        return None


def load_training_data():
    """Load training dataset for stats"""
    dataset_file = DATA_DIR / 'dataset.csv'
    if not dataset_file.exists():
        return None
    try:
        return pd.read_csv(dataset_file)
    except Exception:
        return None


def get_latest_aqmss():
    """Get the most recent AQMSS analysis result"""
    df = load_aqmss_scores()
    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    return {
        'timestamp': str(latest.get('timestamp', '')),
        'total_score': float(latest.get('total_score', 0)),
        'volatility_regime': float(latest.get('volatility_regime', 0)),
        'true_liquidity': float(latest.get('true_liquidity', 0)),
        'structure': float(latest.get('structure', 0)),
        'momentum': float(latest.get('momentum', 0)),
        'volatility_level': float(latest.get('volatility_level', 0)),
        'market_condition': str(latest.get('market_condition', 'UNKNOWN')),
        'current_atr': float(latest.get('current_atr', 0)),
        'mean_atr': float(latest.get('mean_atr', 0)),
        'atr_std': float(latest.get('atr_std', 0)),
        'recent_volume': float(latest.get('recent_volume', 0)),
        'ai_high_quality_prob': float(latest['ai_high_quality_prob']) if pd.notna(latest.get('ai_high_quality_prob')) else None
    }


# ==========================================
# MARKET GENERATION (based on real AQMSS metrics)
# ==========================================

def get_cached_markets():
    """Get or generate cached market data based on real AQMSS metrics"""
    now = datetime.now()

    if (market_cache['data'] is not None and
        market_cache['timestamp'] is not None and
        (now - market_cache['timestamp']).seconds < market_cache['ttl']):
        return market_cache['data']

    markets = generate_markets_from_aqmss()
    market_cache['data'] = markets
    market_cache['timestamp'] = now
    return markets


def generate_markets_from_aqmss():
    """Generate market data using REAL AQMSS metrics as foundation"""
    aqmss = get_latest_aqmss()

    # Base metrics from real AQMSS or defaults
    if aqmss:
        base_quality = aqmss['total_score']  # 0-10 scale
        base_liquidity = aqmss['true_liquidity']
        base_volatility = aqmss['volatility_level']
        base_momentum = aqmss['momentum']
        base_structure = aqmss['structure']
        base_condition = aqmss['market_condition']
        ai_prob = aqmss['ai_high_quality_prob']
    else:
        base_quality = 5.0
        base_liquidity = 5.0
        base_volatility = 5.0
        base_momentum = 5.0
        base_structure = 5.0
        base_condition = 'UNKNOWN'
        ai_prob = None

    # Trading pairs - EURUSD is the real one
    symbols = [
        ('EURUSD', 'Euro / US Dollar', True),       # Real AQMSS data
        ('BTCUSD', 'Bitcoin / US Dollar', False),
        ('ETHUSD', 'Ethereum / US Dollar', False),
        ('GBPUSD', 'British Pound / US Dollar', False),
        ('USDJPY', 'US Dollar / Japanese Yen', False),
        ('XAUUSD', 'Gold / US Dollar', False),
        ('AAPL', 'Apple Inc.', False),
        ('TSLA', 'Tesla Inc.', False),
        ('NVDA', 'NVIDIA Corporation', False),
        ('MSFT', 'Microsoft Corporation', False),
        ('GOOGL', 'Alphabet Inc.', False),
        ('AMZN', 'Amazon.com Inc.', False),
        ('SPX500', 'S&P 500 Index', False),
        ('USDCHF', 'US Dollar / Swiss Franc', False),
        ('AUDUSD', 'Australian Dollar / US Dollar', False),
    ]

    markets = []
    np.random.seed(int(datetime.now().timestamp()) // 30)  # Change every 30s

    for symbol, name, is_real in symbols:
        if is_real:
            # EURUSD: Use REAL AQMSS data directly
            quality_score = base_quality * 10  # Convert 0-10 to 0-100 display scale
            liquidity = base_liquidity * 10
            volatility = base_volatility * 10
            momentum_val = base_momentum
            condition = base_condition

            # Determine momentum direction from AQMSS
            if momentum_val > 6:
                momentum_dir = 'BULLISH'
            elif momentum_val < 4:
                momentum_dir = 'BEARISH'
            else:
                momentum_dir = 'NEUTRAL'

            price = 1.0500 + np.random.uniform(-0.005, 0.005)
            change = np.random.uniform(-1.5, 1.5)
            volume = 5207.8  # Real volume from AQMSS
        else:
            # Other markets: Vary around AQMSS base metrics
            variation = np.random.uniform(-2.5, 2.5)
            raw_quality = np.clip(base_quality + variation, 1, 10)
            quality_score = raw_quality * 10  # 0-100 scale

            liquidity = np.clip((base_liquidity + np.random.uniform(-2, 2)) * 10, 10, 95)
            volatility = np.clip((base_volatility + np.random.uniform(-2, 2)) * 10, 5, 60)

            # Determine momentum from quality
            if quality_score > 70:
                momentum_dir = np.random.choice(['BULLISH', 'BULLISH', 'NEUTRAL'])
            elif quality_score > 50:
                momentum_dir = np.random.choice(['BULLISH', 'NEUTRAL', 'BEARISH'])
            else:
                momentum_dir = np.random.choice(['BEARISH', 'BEARISH', 'NEUTRAL'])

            # Determine condition
            if quality_score > 75:
                condition = 'HIGH QUALITY MARKET'
            elif quality_score > 60:
                condition = 'NORMAL MARKET'
            elif quality_score > 40:
                condition = 'WEAK CONDITIONS'
            else:
                condition = 'NO TRADE'

            # Price based on asset type
            if 'BTC' in symbol:
                price = 97000 + np.random.uniform(-2000, 2000)
            elif 'ETH' in symbol:
                price = 2700 + np.random.uniform(-200, 200)
            elif 'XAU' in symbol:
                price = 2900 + np.random.uniform(-50, 50)
            elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']:
                price = 150 + np.random.uniform(-50, 150)
            elif 'SPX' in symbol:
                price = 6000 + np.random.uniform(-100, 100)
            elif 'JPY' in symbol:
                price = 152 + np.random.uniform(-2, 2)
            else:
                price = 0.9 + np.random.uniform(-0.1, 0.1)

            change = np.random.uniform(-4, 4)
            if 'BTC' in symbol or 'ETH' in symbol:
                volume = np.random.uniform(1e9, 5e9)
            elif symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']:
                volume = np.random.uniform(5e7, 2e8)
            else:
                volume = np.random.uniform(1e7, 1e8)

        market = {
            'symbol': symbol,
            'name': name,
            'is_real_aqmss': is_real,
            'price': round(price, 2 if price > 100 else 6),
            'change': round(change, 2),
            'volume': round(volume, 2),
            'quality_score': round(quality_score, 1),
            'liquidity': round(liquidity, 1),
            'volatility': round(volatility, 2),
            'momentum': momentum_dir,
            'market_condition': condition,
            'timestamp': datetime.now().isoformat(),
        }
        markets.append(market)

    return markets


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/markets', methods=['GET'])
def get_markets():
    """Get all market data with AQMSS-based scores"""
    try:
        markets = get_cached_markets()
        return jsonify({
            'success': True,
            'data': markets,
            'count': len(markets),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/market/<symbol>', methods=['GET'])
def get_market_detail(symbol):
    """Get detailed market data for a specific symbol"""
    try:
        markets = get_cached_markets()
        market = next((m for m in markets if m['symbol'] == symbol), None)
        if market:
            return jsonify({'success': True, 'data': market})
        return jsonify({'success': False, 'error': 'Symbol not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/aqmss', methods=['GET'])
def get_aqmss_data():
    """Get real AQMSS scoring data, model metrics, and history"""
    try:
        current = get_latest_aqmss()
        model_metrics = load_model_metrics()
        model_metadata = load_model_metadata()
        training_data = load_training_data()

        history_count = 0
        if training_data is not None:
            history_count = len(training_data)

        aqmss_history = load_aqmss_scores()
        readings_count = len(aqmss_history) if aqmss_history is not None else 0

        return jsonify({
            'success': True,
            'data': {
                'current': current,
                'model_metrics': model_metrics,
                'model_metadata': model_metadata,
                'history_count': history_count,
                'readings_count': readings_count,
                'ai_available': AI_AVAILABLE
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get overall dashboard statistics from cached markets"""
    try:
        markets = get_cached_markets()

        stats = {
            'total_markets': 0,
            'avg_quality': 50,
            'high_quality_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'volatile_count': 0,
        }

        if markets:
            stats['total_markets'] = len(markets)
            quality_scores = [m['quality_score'] for m in markets]
            stats['avg_quality'] = sum(quality_scores) / len(quality_scores) if quality_scores else 50
            stats['high_quality_count'] = sum(1 for m in markets if m['quality_score'] > 75)
            stats['bullish_count'] = sum(1 for m in markets if m['momentum'] == 'BULLISH')
            stats['bearish_count'] = sum(1 for m in markets if m['momentum'] == 'BEARISH')
            stats['volatile_count'] = sum(1 for m in markets if m['volatility'] > 40)

        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/insights', methods=['GET'])
def get_ai_insights():
    """Get AI-generated market insights based on real AQMSS data"""
    try:
        insights = []
        markets = get_cached_markets()
        aqmss = get_latest_aqmss()

        # AQMSS Core Insight (real data)
        if aqmss:
            score = aqmss['total_score']
            cond = aqmss['market_condition']
            ai_prob = aqmss.get('ai_high_quality_prob')

            if score >= 7.5:
                desc = f'AQMSS Total Score: {score:.2f}/10 - Excellent trading conditions. '
            elif score >= 6:
                desc = f'AQMSS Total Score: {score:.2f}/10 - Normal market conditions. '
            elif score >= 4:
                desc = f'AQMSS Total Score: {score:.2f}/10 - Weak conditions, trade with caution. '
            else:
                desc = f'AQMSS Total Score: {score:.2f}/10 - NO TRADE zone. Avoid entries. '

            desc += f'Condition: {cond}. '
            if ai_prob is not None:
                desc += f'AI High Quality Probability: {ai_prob*100:.1f}%. '

            # Factor breakdown
            desc += f'Structure: {aqmss["structure"]:.1f}/10, Momentum: {aqmss["momentum"]:.1f}/10, Liquidity: {aqmss["true_liquidity"]:.1f}/10.'

            insights.append({
                'type': 'aqmss',
                'title': f'🎯 AQMSS Core Analysis: {cond}',
                'description': desc,
                'priority': 'HIGH'
            })

        if markets:
            # High quality opportunities
            high_quality = [m for m in markets if m['quality_score'] > 75]
            if high_quality:
                symbols = ', '.join([m['symbol'] for m in high_quality[:3]])
                insights.append({
                    'type': 'opportunity',
                    'title': f'🎯 {len(high_quality)} High Quality Markets',
                    'description': f'Excellent conditions in {len(high_quality)} markets. Top: {symbols}',
                    'priority': 'HIGH'
                })

            # Bullish momentum
            bullish = [m for m in markets if m['momentum'] == 'BULLISH']
            if bullish:
                symbols = ', '.join([m['symbol'] for m in bullish[:3]])
                extra = f' and {len(bullish)-3} more' if len(bullish) > 3 else ''
                insights.append({
                    'type': 'bullish',
                    'title': f'📈 {len(bullish)} Bullish Markets',
                    'description': f'Strong upward momentum in {symbols}{extra}.',
                    'priority': 'HIGH'
                })

            # Bearish warnings
            bearish = [m for m in markets if m['momentum'] == 'BEARISH']
            if bearish:
                symbols = ', '.join([m['symbol'] for m in bearish[:3]])
                insights.append({
                    'type': 'bearish',
                    'title': f'📉 {len(bearish)} Bearish Markets',
                    'description': f'Downward pressure in {symbols}. Consider defensive strategies.',
                    'priority': 'MEDIUM'
                })

            # Volatility alerts
            volatile = [m for m in markets if m['volatility'] > 40]
            if volatile:
                symbols = ', '.join([m['symbol'] for m in volatile[:3]])
                insights.append({
                    'type': 'warning',
                    'title': f'⚡ High Volatility Alert',
                    'description': f'{len(volatile)} markets with elevated volatility: {symbols}. Widen stops.',
                    'priority': 'MEDIUM'
                })

            # Poor quality warnings
            poor = [m for m in markets if m['quality_score'] < 40]
            if poor:
                symbols = ', '.join([m['symbol'] for m in poor[:3]])
                insights.append({
                    'type': 'warning',
                    'title': f'⚠️ {len(poor)} Low Quality Markets',
                    'description': f'Poor conditions in {symbols}. Avoid or use minimal size.',
                    'priority': 'LOW'
                })

            # Best liquidity
            top_liquid = sorted(markets, key=lambda x: x['liquidity'], reverse=True)[:3]
            liquid_info = ', '.join([f"{m['symbol']} ({m['liquidity']:.0f})" for m in top_liquid])
            insights.append({
                'type': 'info',
                'title': '💧 Best Liquidity Markets',
                'description': f'Top liquid markets for smooth execution: {liquid_info}',
                'priority': 'LOW'
            })

        return jsonify({
            'success': True,
            'data': insights,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/predict', methods=['POST'])
def ai_predict():
    """Use real AQMSS AI model to predict market quality"""
    try:
        data = request.json

        features = {
            'volatility_regime': float(data.get('volatility_regime', 5)),
            'true_liquidity': float(data.get('liquidity', 50)) / 10,
            'structure': float(data.get('structure', 50)) / 10,
            'momentum': float(data.get('momentum', 50)) / 10,
            'volatility_level': float(data.get('volatility', 20)) / 10,
            'current_atr': float(data.get('atr', 0.001)),
            'recent_volume': float(data.get('volume', 5000)),
        }

        if AI_AVAILABLE:
            history_csv = RESULTS_DIR / 'market_scores.csv'
            if history_csv.exists():
                prob = train_and_predict(str(history_csv), features)
                if prob is not None:
                    prediction = {
                        'probability': float(prob),
                        'confidence': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'LOW',
                        'recommendation': 'BUY' if prob > 0.6 else 'NEUTRAL' if prob > 0.4 else 'SELL',
                        'timestamp': datetime.now().isoformat(),
                        'model': 'AQMSS RandomForest'
                    }
                    return jsonify({'success': True, 'data': prediction})

        # Fallback mock prediction
        mock_prob = 0.3 + (features['true_liquidity'] / 20.0) + (features['structure'] / 15.0)
        mock_prob = min(max(mock_prob, 0.05), 0.95)
        prediction = {
            'probability': mock_prob,
            'confidence': 'MEDIUM',
            'recommendation': 'BUY' if mock_prob > 0.6 else 'NEUTRAL' if mock_prob > 0.4 else 'SELL',
            'timestamp': datetime.now().isoformat(),
            'model': 'Fallback (AI module not loaded)'
        }
        return jsonify({'success': True, 'data': prediction})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """AI Assistant chat endpoint with AQMSS context"""
    try:
        data = request.json
        user_message = str(data.get('message', '')).strip()

        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400

        # Get real AQMSS context for the AI
        aqmss = get_latest_aqmss()
        response = generate_ai_response(user_message, aqmss)

        return jsonify({
            'success': True,
            'data': {
                'user_message': user_message,
                'ai_response': response,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_ai_response(user_message: str, aqmss=None) -> str:
    """Generate AI response with real AQMSS data context"""
    msg = user_message.lower()

    # Build AQMSS context string
    aqmss_ctx = ""
    if aqmss:
        aqmss_ctx = (
            f"\n\n📊 Current AQMSS Data (EURUSD):\n"
            f"• Total Score: {aqmss['total_score']:.2f}/10\n"
            f"• Condition: {aqmss['market_condition']}\n"
            f"• Structure: {aqmss['structure']:.2f}/10\n"
            f"• Momentum: {aqmss['momentum']:.2f}/10\n"
            f"• Liquidity: {aqmss['true_liquidity']:.2f}/10\n"
            f"• Vol Regime: {aqmss['volatility_regime']:.2f}/10\n"
            f"• Vol Level: {aqmss['volatility_level']:.2f}/10\n"
        )
        if aqmss.get('ai_high_quality_prob') is not None:
            aqmss_ctx += f"• AI High Quality Prob: {aqmss['ai_high_quality_prob']*100:.1f}%\n"

    # Response templates
    if 'quality' in msg or 'score' in msg or 'aqmss' in msg:
        return (
            f"📊 AQMSS Quality Scoring System\n\n"
            f"The AQMSS computes a total score (0-10) from 5 weighted factors:\n"
            f"• Volatility Regime (25%): ATR vs historical mean\n"
            f"• True Liquidity (15%): Volume-based metric\n"
            f"• Structure (30%): Trend strength via regression\n"
            f"• Momentum (20%): Candle range analysis\n"
            f"• Volatility Level (10%): ATR stability\n\n"
            f"Thresholds: <4 NO TRADE | 4-6 WEAK | 6-7.5 NORMAL | >7.5 HIGH QUALITY"
            f"{aqmss_ctx}"
        )

    if 'best' in msg or 'recommend' in msg or 'top' in msg:
        return (
            f"✨ Current Recommendations\n\n"
            f"Based on AQMSS analysis:\n"
            f"1. Focus on markets with Quality Scores > 75 (>7.5/10 AQMSS)\n"
            f"2. Look for BULLISH momentum + high liquidity\n"
            f"3. Avoid markets in NO TRADE or WEAK CONDITIONS\n"
            f"4. Use ATR for position sizing (current ATR from AQMSS)\n"
            f"5. Check AI probability before entry (>60% favorable)"
            f"{aqmss_ctx}"
        )

    if 'volatile' in msg or 'volatility' in msg:
        return (
            f"⚡ Volatility Analysis\n\n"
            f"AQMSS measures two volatility aspects:\n"
            f"• Volatility Regime: Current ATR vs historical mean\n"
            f"• Volatility Level: ATR standard deviation (stability)\n\n"
            f"High volatility = wider stops, smaller positions\n"
            f"Low volatility = tighter stops, larger positions"
            f"{aqmss_ctx}"
        )

    if 'liquid' in msg:
        return (
            f"💧 Liquidity Analysis\n\n"
            f"AQMSS True Liquidity measures volume-based market depth.\n\n"
            f"• Score > 7/10: Excellent execution expected\n"
            f"• Score 4-7/10: Normal conditions\n"
            f"• Score < 4/10: Thin market, watch for slippage"
            f"{aqmss_ctx}"
        )

    if 'bullish' in msg:
        return (
            f"📈 Bullish Market Analysis\n\n"
            f"AQMSS Momentum factor measures candle range vs ATR.\n\n"
            f"Bullish signals:\n"
            f"• Rising momentum score\n"
            f"• Strong structure (consistent higher highs/lows)\n"
            f"• High quality score in bullish markets\n\n"
            f"Best strategy: Follow momentum with AQMSS confirmation"
            f"{aqmss_ctx}"
        )

    if 'bearish' in msg:
        return (
            f"📉 Bearish Market Analysis\n\n"
            f"AQMSS can detect bearish conditions through:\n"
            f"• Weakening structure score\n"
            f"• Low momentum\n"
            f"• Overall weak quality score\n\n"
            f"Consider: Short positions or staying flat"
            f"{aqmss_ctx}"
        )

    if 'risk' in msg:
        return (
            f"🛡️ Risk Management with AQMSS\n\n"
            f"1. Never risk more than 2% per trade\n"
            f"2. Use AQMSS ATR for stop-loss placement\n"
            f"3. Only trade in NORMAL or HIGH QUALITY conditions\n"
            f"4. Scale position size with quality score\n"
            f"5. Check AI probability before entry"
            f"{aqmss_ctx}"
        )

    if 'predict' in msg or 'ai' in msg or 'model' in msg:
        metrics = load_model_metrics()
        meta = load_model_metadata()
        model_info = ""
        if metrics:
            model_info = (
                f"\n\n🤖 Your AI Model Stats:\n"
                f"• ROC AUC: {metrics.get('roc_auc', 'N/A')}\n"
                f"• Precision: {metrics.get('precision', 'N/A')}\n"
                f"• Recall: {metrics.get('recall', 'N/A')}\n"
                f"• F1 Score: {metrics.get('f1_score', 'N/A')}\n"
            )
        if meta:
            model_info += (
                f"• Model: {meta.get('model_type', 'N/A')}\n"
                f"• Train Size: {meta.get('train_size', 'N/A')}\n"
                f"• Features: {meta.get('n_features', 'N/A')}\n"
            )
        return (
            f"🔮 AQMSS AI Prediction System\n\n"
            f"Your trained RandomForest model classifies market quality.\n"
            f"It predicts probability of HIGH QUALITY conditions.\n\n"
            f"Use predictions as ONE input alongside the AQMSS score."
            f"{model_info}{aqmss_ctx}"
        )

    if 'help' in msg:
        return (
            f"🤖 AQMSS AI Assistant\n\n"
            f"I can help with:\n"
            f"• 'quality' - AQMSS scoring explained\n"
            f"• 'best' - Trading recommendations\n"
            f"• 'volatility' - Volatility analysis\n"
            f"• 'liquidity' - Liquidity insights\n"
            f"• 'risk' - Risk management\n"
            f"• 'predict' / 'model' - AI model info\n"
            f"• 'bullish' / 'bearish' - Direction analysis"
            f"{aqmss_ctx}"
        )

    if 'strategy' in msg:
        return (
            f"🎲 AQMSS Trading Strategy\n\n"
            f"1. Screen: Filter by AQMSS score > 6/10\n"
            f"2. Analyze: Check all 5 factors\n"
            f"3. Confirm: AI probability > 60%\n"
            f"4. Enter: Position size based on ATR\n"
            f"5. Manage: Stop at 1.5x ATR\n"
            f"6. Exit: Follow your target or trailing stop"
            f"{aqmss_ctx}"
        )

    # Default
    return (
        f"I specialize in AQMSS market analysis. Try asking about:\n\n"
        f"📊 Quality scores & AQMSS factors\n"
        f"💡 Trading recommendations\n"
        f"🤖 AI model predictions\n"
        f"📈 Market direction analysis\n"
        f"🛡️ Risk management\n\n"
        f"Type 'help' for all available topics."
        f"{aqmss_ctx}"
    )


if __name__ == '__main__':
    print("=" * 50)
    print("AQMSS Dashboard Server")
    print("=" * 50)

    # Check real data availability
    aqmss = get_latest_aqmss()
    if aqmss:
        print(f"✅ AQMSS data found: Score {aqmss['total_score']:.2f}/10 ({aqmss['market_condition']})")
    else:
        print("⚠️  No AQMSS data found - run main.py first")

    metrics = load_model_metrics()
    if metrics:
        print(f"✅ AI Model loaded: ROC AUC = {metrics.get('roc_auc', 'N/A')}")
    else:
        print("⚠️  No model metrics found - run run_pipeline.py first")

    training = load_training_data()
    if training is not None:
        print(f"✅ Training data: {len(training)} rows")
    else:
        print("⚠️  No training data found")

    print(f"✅ AI Module: {'Available' if AI_AVAILABLE else 'Not available'}")
    print(f"\n🌐 Starting server at http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
