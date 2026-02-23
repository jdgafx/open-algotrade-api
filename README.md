# Open Algotrade API

Algorithmic Trading SaaS Backend - FastAPI + Hyperliquid DEX integration.

## Deploy

### Render (Recommended)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/jdgafx/open-algotrade-api)

### Railway
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template?referralCode=open-algotrade&repo=https://github.com/jdgafx/open-algotrade-api)

### Docker
```bash
docker build -t open-algotrade-api .
docker run -p 8000:8000 --env-file .env open-algotrade-api
```

### Local
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/users` | Create user |
| GET | `/users/{id}` | Get user |
| GET | `/vault/status` | Vault equity, NAV, shares |
| POST | `/deposit` | Deposit USD (min $0.25) |
| POST | `/withdraw` | Withdraw shares |
| GET | `/portfolio/{id}` | User portfolio value |
| GET | `/positions` | Live Hyperliquid positions |
| GET | `/trades` | Trade history |
| GET | `/strategy/status` | Strategy state |
| POST | `/strategy/start` | Start turtle strategy |
| POST | `/strategy/stop` | Stop strategy |
| GET | `/market/price/{symbol}` | Live market price |

## Architecture

- **Frontend**: Next.js 15 + Shadboard → [hyperliquid-trading-saas.netlify.app](https://hyperliquid-trading-saas.netlify.app)
- **Backend**: FastAPI + SQLAlchemy (this repo)
- **Trading**: Hyperliquid DEX via `hyperliquid-python-sdk`
- **Strategy**: Turtle Trading (55-bar breakout, ATR stops)
- **Vault**: Pooled architecture for $0.25 minimum investments

## Environment Variables

```
HYPERLIQUID_PRIVATE_KEY=   # Your Hyperliquid wallet private key
HYPERLIQUID_NETWORK=testnet # testnet or mainnet
VAULT_ADDRESS=              # Optional: pre-existing vault address
```
