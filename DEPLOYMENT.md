# 🚀 Deployment Guide

Panduan untuk deploy Dashboard TKDD & Kemiskinan ke berbagai platform.

## 📋 Pre-Deployment Checklist

- [ ] Semua dependencies ada di `requirements.txt`
- [ ] Code sudah tested di local
- [ ] Data sudah disiapkan atau ada mekanisme upload
- [ ] Dokumentasi lengkap (README.md)
- [ ] Environment variables (jika ada) sudah dikonfigurasi

---

## 🌐 Option 1: Streamlit Cloud (Recommended - FREE)

**Pros:**
- ✅ Gratis untuk public apps
- ✅ Setup mudah dan cepat
- ✅ Auto-deploy dari GitHub
- ✅ Built-in resource management
- ✅ HTTPS otomatis

**Cons:**
- ❌ Resource terbatas (1 GB RAM)
- ❌ Apps bisa sleep jika tidak digunakan

### Langkah Deployment:

#### 1. Push ke GitHub
```bash
# Initialize git (jika belum)
git init
git add .
git commit -m "Initial commit - TKDD Dashboard"

# Create repo di GitHub, lalu:
git remote add origin https://github.com/username/dashboard-tkdd.git
git branch -M main
git push -u origin main
```

#### 2. Deploy di Streamlit Cloud

1. **Kunjungi:** https://share.streamlit.io
2. **Sign in** dengan GitHub
3. **New app** → Pilih repository Anda
4. **Configure:**
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: 3.9+
5. **Deploy!**

#### 3. (Opsional) Custom Domain
- Settings → Custom domains
- Tambahkan CNAME record di DNS provider

**URL Format:** `https://username-dashboard-tkdd-app-xxxxx.streamlit.app`

---

## 🐳 Option 2: Docker + Any Cloud Platform

**Pros:**
- ✅ Portable dan consistent
- ✅ Kontrol penuh atas environment
- ✅ Bisa deploy di mana saja

### Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Create .dockerignore

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.env
.git
.gitignore
*.md
*.txt
!requirements.txt
Data/
Screenshots/
```

### Build dan Run

```bash
# Build image
docker build -t tkdd-dashboard .

# Run container
docker run -p 8501:8501 tkdd-dashboard

# Test
curl http://localhost:8501
```

### Deploy ke Cloud

#### Google Cloud Run
```bash
# Install gcloud CLI
gcloud auth login

# Build dan push
gcloud builds submit --tag gcr.io/PROJECT_ID/tkdd-dashboard

# Deploy
gcloud run deploy tkdd-dashboard \
  --image gcr.io/PROJECT_ID/tkdd-dashboard \
  --platform managed \
  --region asia-southeast2 \
  --allow-unauthenticated \
  --memory 2Gi
```

#### AWS ECS
```bash
# Build dan push ke ECR
aws ecr create-repository --repository-name tkdd-dashboard
docker tag tkdd-dashboard:latest AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/tkdd-dashboard
docker push AWS_ACCOUNT.dkr.ecr.REGION.amazonaws.com/tkdd-dashboard

# Deploy via ECS Console atau CLI
```

#### Azure Container Instances
```bash
az login

# Create resource group
az group create --name tkdd-dashboard-rg --location southeastasia

# Deploy container
az container create \
  --resource-group tkdd-dashboard-rg \
  --name tkdd-dashboard \
  --image tkdd-dashboard:latest \
  --dns-name-label tkdd-dashboard \
  --ports 8501
```

---

## 🖥️ Option 3: Virtual Private Server (VPS)

**Platforms:** DigitalOcean, Linode, Vultr, AWS EC2

**Pros:**
- ✅ Kontrol penuh
- ✅ Bisa handle large datasets
- ✅ Custom configuration

**Cons:**
- ❌ Perlu maintenance
- ❌ Lebih mahal untuk high availability

### Setup di Ubuntu VPS

#### 1. Connect to VPS
```bash
ssh user@your-vps-ip
```

#### 2. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dan pip
sudo apt install python3-pip python3-venv -y

# Install git
sudo apt install git -y
```

#### 3. Clone dan Setup Project
```bash
# Clone repository
git clone https://github.com/username/dashboard-tkdd.git
cd dashboard-tkdd

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 4. Run Dashboard

**Option A: Simple Run (for testing)**
```bash
streamlit run app.py --server.port 8501
```

**Option B: Production with Nginx + Supervisor**

**Install Nginx:**
```bash
sudo apt install nginx -y
```

**Configure Nginx** (`/etc/nginx/sites-available/tkdd-dashboard`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/tkdd-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Install Supervisor:**
```bash
sudo apt install supervisor -y
```

**Configure Supervisor** (`/etc/supervisor/conf.d/tkdd-dashboard.conf`):
```ini
[program:tkdd-dashboard]
directory=/home/user/dashboard-tkdd
command=/home/user/dashboard-tkdd/venv/bin/streamlit run app.py --server.port 8501
user=user
autostart=true
autorestart=true
stderr_logfile=/var/log/tkdd-dashboard.err.log
stdout_logfile=/var/log/tkdd-dashboard.out.log
```

**Start service:**
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start tkdd-dashboard
```

#### 5. Setup SSL (HTTPS)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal test
sudo certbot renew --dry-run
```

---

## 🌍 Option 4: Heroku

**Pros:**
- ✅ Mudah untuk deploy
- ✅ Free tier available
- ✅ Git-based deployment

**Cons:**
- ❌ Free tier terbatas (apps sleep after 30 min inactivity)

### Setup Files

#### Procfile
```
web: sh setup.sh && streamlit run app.py
```

#### setup.sh
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

#### runtime.txt
```
python-3.9.16
```

### Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create tkdd-dashboard

# Deploy
git push heroku main

# Open app
heroku open
```

---

## 📊 Option 5: Hugging Face Spaces

**Pros:**
- ✅ Free untuk public apps
- ✅ ML community friendly
- ✅ Good for demo apps

### Setup

1. Create account di https://huggingface.co
2. Create new Space (type: Streamlit)
3. Upload files atau connect GitHub
4. Configure `README.md`:

```yaml
---
title: Dashboard TKDD & Kemiskinan
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---
```

---

## 🔒 Security Best Practices

### 1. Environment Variables

**Jangan hardcode credentials!**

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

**Add to .gitignore:**
```
.env
```

### 2. Input Validation

```python
def validate_csv(df):
    required_columns = ['Tahun', 'Kabupaten/kota', ...]
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns")
    
    return True
```

### 3. Rate Limiting

```python
import streamlit as st
import time

if 'last_run' not in st.session_state:
    st.session_state.last_run = 0

current_time = time.time()
if current_time - st.session_state.last_run < 5:  # 5 seconds cooldown
    st.warning("Please wait before running again")
    st.stop()

st.session_state.last_run = current_time
```

### 4. HTTPS Only

- Selalu gunakan HTTPS di production
- Gunakan SSL certificates (Let's Encrypt gratis)
- Set secure headers di web server

---

## 📈 Performance Optimization

### 1. Caching

```python
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')
```

### 2. Lazy Loading

```python
# Load data only when needed
if st.button('Load Data'):
    data = load_large_dataset()
```

### 3. Reduce Memory Usage

```python
# Use appropriate dtypes
df['year'] = df['year'].astype('int16')
df['category'] = df['category'].astype('category')
```

### 4. Async Operations

```python
import asyncio

async def process_data():
    # Long-running operation
    pass
```

---

## 🔧 Maintenance

### Monitoring

**Option 1: Built-in Streamlit Metrics**
```python
import streamlit as st

# Check if app is healthy
st._config.get_option('server.enableWebsocketCompression')
```

**Option 2: External Monitoring**
- UptimeRobot
- Pingdom
- New Relic
- Datadog

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# In your code
logger.info("Data loaded successfully")
logger.error(f"Error processing: {str(e)}")
```

### Updates

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart app (if using supervisor)
sudo supervisorctl restart tkdd-dashboard
```

---

## 💰 Cost Estimation

### Free Options:
- **Streamlit Cloud:** Free (1 GB RAM, public apps)
- **Heroku:** Free tier (550 hours/month)
- **Hugging Face:** Free (2 CPU cores, 16 GB disk)

### Paid Options:
- **VPS (DigitalOcean):** $4-12/month (1-2 GB RAM)
- **Google Cloud Run:** Pay-per-use (~$5-20/month untuk small apps)
- **AWS EC2:** $8-20/month (t2.micro - t2.small)
- **Streamlit Cloud (Team):** $250/month

---

## 🆘 Troubleshooting

### App Won't Start

**Check logs:**
```bash
# Streamlit Cloud: View logs in dashboard
# Heroku: heroku logs --tail
# VPS: sudo tail -f /var/log/tkdd-dashboard.err.log
```

### Memory Issues

**Solution:**
- Reduce dataset size
- Use sampling
- Upgrade to higher tier
- Implement pagination

### Slow Performance

**Solution:**
- Add caching (`@st.cache_data`)
- Optimize queries
- Reduce plot complexity
- Use CDN for static files

---

## 📞 Support & Resources

**Streamlit:**
- Docs: https://docs.streamlit.io
- Forum: https://discuss.streamlit.io
- GitHub: https://github.com/streamlit/streamlit

**Deployment Platforms:**
- Streamlit Cloud: https://streamlit.io/cloud
- Heroku: https://www.heroku.com
- DigitalOcean: https://www.digitalocean.com
- Google Cloud: https://cloud.google.com
- Hugging Face: https://huggingface.co/spaces

---

**Happy Deploying! 🚀✨**

**Recommended for beginners:** Start with Streamlit Cloud (easiest, free)
**Recommended for production:** VPS with Nginx + Supervisor (full control)
**Recommended for enterprise:** Google Cloud Run or AWS ECS (scalable)
