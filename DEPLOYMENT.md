# 🚀 Face Recognition API Deployment Guide

## Overview

This face recognition attendance system can be deployed on three platforms:
- **Docker** - Full-featured deployment with enhanced face detection
- **Render** - Cloud platform with persistent servers for ML workloads
- **Vercel** - Serverless deployment with lightweight face detection

---

## 🐳 Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available

### Step 1: Prepare Files
```bash
# Clone/download the project
cd face-recognition-api

# Ensure training data is in place
ls train/  # Should contain student folders (AD002, AD004, etc.)
```

### Step 2: Build and Run
```bash
# Build and start the container
docker-compose up --build -d

# Check logs
docker logs api-face-recognition-1

# Test the deployment
curl http://localhost:8080/health
```

### Step 3: Train the Model
```bash
# Train with your data
curl -X POST http://localhost:8080/train

# Expected response: {"status":"ok","updated":X,"total_students":Y}
```

### Step 4: Test Face Recognition
```bash
# Upload an image for recognition
curl -X POST -F "file=@test_image.jpg" http://localhost:8080/recognize
```

### Docker Features
- ✅ Enhanced face detection (reduced skipped faces from 77→0)
- ✅ Full InsightFace ML pipeline
- ✅ Persistent data storage via volumes
- ✅ Production-ready error handling
- ✅ GPU support ready (uncomment CUDA parts if needed)

---

## 🟢 Render Deployment

### Prerequisites
- Git repository (GitHub/GitLab)
- Render.com account

### Step 1: Prepare Repository
```bash
# Initialize git if needed
git init
git add .
git commit -m "Initial commit"

# Push to GitHub/GitLab
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: Create Render Service
1. Go to [Render.com](https://render.com) and sign in
2. Click "New +" → "Web Service"
3. Connect your GitHub/GitLab repository
4. Choose the repository containing this project

### Step 3: Configure Service
```yaml
# Use these settings (already in render.yaml):
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300

# Environment Variables:
RENDER=true
FLASK_ENV=production
PYTHONUNBUFFERED=true
```

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait for build to complete (5-10 minutes)
3. Test at: `https://your-service-name.onrender.com/health`

### Step 5: Upload Training Data
Since Render doesn't have access to your local `train/` folder:
1. Create a ZIP of your training data
2. Use a cloud storage service (Google Drive, S3, etc.)
3. Update the code to download training data, or
4. Use the manual attendance marking feature

### Render Features
- ✅ Full ML pipeline with InsightFace
- ✅ 300-second timeouts for training
- ✅ Persistent servers (no cold starts)
- ✅ Automatic deployments from Git
- ✅ Free tier: 750 hours/month

---

## 🔵 Vercel Deployment (Lightweight)

### ⚠️ **Important: Limited Features**
Vercel has strict serverless limitations. This deployment only supports **manual attendance marking**.

### Prerequisites
- Node.js and npm installed
- Vercel account

### Step 1: Prepare for Vercel
```bash
# Install Vercel CLI
npm install -g vercel

# Copy lightweight requirements
cp requirements-vercel.txt requirements.txt
```

### Step 2: Deploy to Vercel
```bash
# Deploy (vercel.json and api/index.py are already configured)
vercel --prod

# Follow prompts:
# - Link to existing project? No
# - Project name: your-project-name
# - Directory: ./
# - Settings? No
```

### Step 3: Test Deployment
```bash
# Test health check
curl https://your-app.vercel.app/health

# Test manual attendance
curl -X POST https://your-app.vercel.app/mark_manual \
  -H "Content-Type: application/json" \
  -d '{"rolls":[1,2,3],"section":"A"}'
```

### Vercel Features
- ✅ **Manual attendance marking** - Fully functional
- ✅ **Global CDN** - Fast worldwide access
- ✅ **Auto-scaling** - Handle traffic spikes
- ✅ **Free tier** - 100GB bandwidth/month
- ❌ **Face recognition disabled** - Package too large
- ❌ **Training disabled** - Serverless limitations

### Why Limited?
- **50MB limit**: ML libraries exceed Vercel's size limit
- **10s timeout**: Model loading takes too long
- **No storage**: Can't save training data or models

### **Recommendation**: Use Vercel for demos/manual attendance, Docker/Render for full ML features

---

## 📊 Platform Comparison

| Feature | Docker | Render | Vercel |
|---------|--------|--------|---------|
| **Setup Complexity** | Medium | Easy | Easy |
| **ML Capabilities** | Full | Full | Limited |
| **Training Support** | ✅ Full | ✅ Full | ❌ Limited |
| **Face Detection** | Enhanced | Standard | Basic |
| **Cost (Free Tier)** | Infrastructure | 750hrs/month | 100GB/month |
| **Scaling** | Manual | Auto (paid) | Auto |
| **Best For** | Production/On-premise | Cloud ML | Demos/API |

---

## 🧪 Testing Your Deployment

### Health Check
```bash
curl https://your-deployment-url/health
# Expected: {"status":"healthy","platform":"..."}
```

### Manual Attendance
```bash
curl -X POST https://your-deployment-url/mark_manual \
  -H "Content-Type: application/json" \
  -d '{"rolls":[1,2,3],"section":"A"}'
```

### Face Recognition (if supported)
```bash
curl -X POST https://your-deployment-url/recognize \
  -F "file=@test_image.jpg"
```

---

## 🔧 Troubleshooting

### Common Issues

**Docker:**
- Port already in use: Change port in docker-compose.yml
- Permission errors: Check volume mount permissions
- CUDA errors: Normal, app falls back to CPU

**Render:**
- Build timeout: Increase timeout in render.yaml
- Memory errors: Upgrade plan or reduce workers
- No training data: Upload via cloud storage

**Vercel:**
- Package too large: Remove heavy dependencies
- Timeout errors: Reduce operation complexity
- Import errors: Check serverless compatibility

### Getting Help
1. Check platform logs
2. Test locally first
3. Verify environment variables
4. Check file permissions

---

## 🎯 Recommendations

### Start Here:
1. **Docker** - If you have technical expertise and want full control
2. **Render** - If you want easy cloud deployment with full ML features
3. **Vercel** - If you need a quick demo or simple recognition API

### Production:
- Use **Docker** for on-premise or custom infrastructure
- Use **Render** for cloud-based ML workloads
- Use **Vercel** for global edge deployment of simple services

The face recognition system will automatically adapt its features based on the deployment platform! 🚀