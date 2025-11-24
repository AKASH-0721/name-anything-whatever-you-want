#!/bin/bash
# Fixed Vercel deployment script

echo "🔧 Preparing fixed Vercel deployment..."

# Copy lightweight requirements
echo "📦 Using ultra-lightweight requirements for Vercel..."
cp requirements-vercel.txt requirements.txt

# Verify required files exist
echo "✅ Checking Vercel configuration files..."
if [ ! -f "vercel.json" ]; then
    echo "❌ vercel.json not found!"
    exit 1
fi

if [ ! -f "api/index.py" ]; then
    echo "❌ api/index.py not found!"
    exit 1
fi

# Show what we're deploying
echo "📋 Vercel deployment will include:"
echo "   - Manual attendance marking (✅ Working)"
echo "   - Health check endpoint (✅ Working)"  
echo "   - Face recognition (❌ Disabled - too large for Vercel)"
echo "   - Training (❌ Disabled - too large for Vercel)"

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
if command -v vercel &> /dev/null; then
    vercel --prod
else
    echo "❌ Vercel CLI not found. Install with: npm install -g vercel"
    exit 1
fi

echo "✅ Vercel deployment completed!"
echo "📝 Test your deployment:"
echo "   curl https://your-app.vercel.app/health"
echo "   curl -X POST https://your-app.vercel.app/mark_manual -H 'Content-Type: application/json' -d '{\"rolls\":[1,2,3]}'"

# Restore original requirements for other platforms
echo "🔄 Restoring full requirements.txt for Docker/Render..."
git checkout requirements.txt

echo "🎉 Vercel deployment ready!"
echo "💡 For full ML features, use Docker or Render deployment instead."