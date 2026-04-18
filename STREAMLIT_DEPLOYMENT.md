# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Create Streamlit Account
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with GitHub account
- Authorize Streamlit to access your repositories

### 2. Deploy the App
1. Click "New app"
2. Select:
   - **Repository:** `CosmicGalactus/real-estate-ml`
   - **Branch:** `main`
   - **Main file path:** `real-estate-ml/app.py`
3. Click "Deploy"

### 3. Configuration
Streamlit automatically uses settings from `.streamlit/config.toml`:
- Theme: Light (professional appearance)
- Layout: Wide (better for charts)
- Server settings optimized for performance

### 4. Wait for Build
- First build takes 2-5 minutes
- You'll see build logs in real-time
- Once complete, get a shareable URL like: `https://yourusername-real-estate-ml-xxxxx.streamlit.app`

---

## What Gets Deployed

✅ **Tab 1: Price Prediction** (Milestone 1)
- ML model predictions
- Model performance metrics

✅ **Tab 2: Model Performance** (Milestone 1)
- Accuracy, precision, recall
- Confusion matrix
- Feature importance

✅ **Tab 3: AI Advisory** (Milestone 2)
- Property input form
- AI analysis and recommendations
- JSON/text report export

✅ **Tab 4: How It Works** (Milestone 2)
- System explanation
- Analysis process overview
- Tips for best results

✅ **Tab 5: About** (Milestone 1)
- Project information
- Contact & details

---

## Environment Variables (Optional)

If you add OpenAI API integration in the future:

1. Go to app settings in Streamlit Cloud
2. Add secrets:
   ```
   OPENAI_API_KEY = "your-key-here"
   ```

For this deployment, no external API keys needed (using mock data).

---

## Testing the Deployed App

After deployment completes:

1. **Price Prediction Tab:** Try predicting a property price
2. **AI Advisory Tab:** Enter property details and generate recommendations
3. **Export:** Download JSON/text reports
4. **Performance:** Check model metrics

---

## Troubleshooting

**App fails to load:**
- Check `.streamlit/config.toml` exists
- Verify all imports work locally: `python3 -c "import streamlit; print('OK')"`

**Missing dependencies:**
- Ensure all packages in `requirements.txt` are compatible
- Check Streamlit Cloud build logs for specific errors

**Chroma database errors:**
- Chroma is included in `requirements.txt`
- Database files are stored in `chroma_db/` directory

---

## Public URL

Once deployed, share the URL:
```
https://yourusername-real-estate-ml-xxxxx.streamlit.app
```

This is what you provide for evaluation!

---

## Redeploying Updates

After making changes locally:

1. Commit and push to GitHub:
   ```bash
   git add -A
   git commit -m "Update message"
   git push origin main
   ```

2. Streamlit Cloud automatically redeploys within 5 minutes

No manual redeployment needed - it watches your GitHub repo!

---

## Deployment Status

- ✅ Code ready
- ✅ Dependencies configured
- ✅ GitHub repository synchronized
- ⏳ Ready for Streamlit Cloud deployment

**Next Step:** Go to share.streamlit.io and follow steps 1-2 above!
