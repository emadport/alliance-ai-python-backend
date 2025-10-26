# Alliance AI Backend

FastAPI backend for ML model training and predictions.

## Quick Deploy to Railway

### 1. Create GitHub repo

```bash
git init
git add .
git commit -m "Initial backend"
git remote add origin https://github.com/YOUR_USERNAME/allianceai-backend.git
git push -u origin main
```

### 2. Deploy to Railway

1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `allianceai-backend` repo
5. Railway auto-detects Python and deploys!

### 3. Get your URL

- Railway will give you a URL like: `https://your-app.railway.app`
- Copy this URL

### 4. Add to Vercel

Go to your Next.js Vercel project → Settings → Environment Variables

- Key: `NEXT_PUBLIC_PYTHON_API_URL`
- Value: `https://your-app.railway.app`

Done! 🎉
