const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const stringSimilarity = require('string-similarity');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;

// Fixed model configuration
// Using gemini-pro as it's the most widely available and least expensive
const MODEL_NAME = process.env.MODEL_NAME || 'gemini-pro';
const TEMPERATURE = parseFloat(process.env.TEMPERATURE) || 0.0;

// CORS configuration
const allowedOrigins = process.env.FRONTEND_ORIGIN
  ? process.env.FRONTEND_ORIGIN.split(',').map((origin) => origin.trim())
  : ['http://localhost:3000', 'http://localhost:5173'];

app.use(
  cors({
    origin: (origin, callback) => {
      if (!origin || allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
        callback(null, true);
      } else {
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
  })
);
app.use(express.json());

// Initialize Gemini AI
let genAI;
if (process.env.GEMINI_API_KEY) {
  genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
}

/**
 * Tokenize text by splitting on whitespace
 */
function tokenize(text) {
  return text
    .toLowerCase()
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

/**
 * Calculate Levenshtein distance between two strings
 */
function levenshteinDistance(str1, str2) {
  const len1 = str1.length;
  const len2 = str2.length;
  const matrix = [];

  for (let i = 0; i <= len1; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= len2; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      if (str1[i - 1] === str2[j - 1]) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j - 1] + 1
        );
      }
    }
  }

  return matrix[len1][len2];
}

/**
 * Check if two strings match using fuzzy logic
 */
function fuzzyMatch(str1, str2) {
  const s1 = str1.toLowerCase().trim();
  const s2 = str2.toLowerCase().trim();

  // Exact match (case-insensitive)
  if (s1 === s2) {
    return true;
  }

  // Substring match
  if (s1.includes(s2) || s2.includes(s1)) {
    return true;
  }

  // Levenshtein distance check
  const distance = levenshteinDistance(s1, s2);
  const maxLen = Math.max(s1.length, s2.length);
  const relativeDistance = maxLen > 0 ? distance / maxLen : 1;

  // Match if Levenshtein distance ≤ 2 OR relative distance ≤ 0.3
  if (distance <= 2 || relativeDistance <= 0.3) {
    return true;
  }

  // String similarity check (using string-similarity library)
  const similarity = stringSimilarity.compareTwoStrings(s1, s2);
  if (similarity >= 0.7) {
    return true;
  }

  return false;
}

/**
 * Find brand mention position in text
 */
function findBrandMention(text, brand) {
  if (!text || !brand) {
    return { mentioned: false, position: 0 };
  }

  const tokens = tokenize(text);
  const brandTokens = tokenize(brand);
  const brandLower = brand.toLowerCase().trim();

  // Check for exact brand name match (case-insensitive)
  const textLower = text.toLowerCase();

  // Try to find brand as a phrase first
  if (textLower.includes(brandLower)) {
    // Find the position of the first token that contains the brand
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (token.includes(brandLower) || brandLower.includes(token)) {
        return { mentioned: true, position: i + 1 };
      }
    }
  }

  // Check each token for fuzzy match
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];

    // Check if token matches any brand token
    for (const brandToken of brandTokens) {
      if (fuzzyMatch(token, brandToken)) {
        return { mentioned: true, position: i + 1 };
      }
    }

    // Check if token matches full brand name
    if (fuzzyMatch(token, brandLower)) {
      return { mentioned: true, position: i + 1 };
    }
  }

  // Check for substring match in full text
  if (textLower.includes(brandLower)) {
    // Find approximate position
    const index = textLower.indexOf(brandLower);
    const beforeText = textLower.substring(0, index);
    const beforeTokens = tokenize(beforeText);
    return { mentioned: true, position: beforeTokens.length + 1 };
  }

  return { mentioned: false, position: 0 };
}

// Health check endpoint
app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Gemini Brand Mention Checker API',
    model: MODEL_NAME,
    temperature: TEMPERATURE,
  });
});

// Main API endpoint
app.post('/api/check', async (req, res) => {
  const { prompt, brand } = req.body;

  if (!prompt || !brand) {
    return res.status(400).json({
      error: 'Both prompt and brand are required',
    });
  }

  // Fallback response if API key is not configured
  if (!genAI || !process.env.GEMINI_API_KEY) {
    return res.json({
      prompt,
      brand,
      mentioned: false,
      position: 0,
      raw: 'API_ERROR',
      error: 'GEMINI_API_KEY not configured on server',
    });
  }

  try {
    const model = genAI.getGenerativeModel({
      model: MODEL_NAME,
      generationConfig: {
        temperature: TEMPERATURE,
        maxOutputTokens: 2048,
      },
    });

    const result = await model.generateContent(prompt);
    const responseText = result.response.text();

    const { mentioned, position } = findBrandMention(responseText, brand);

    return res.json({
      prompt,
      brand,
      mentioned,
      position,
      raw: responseText,
    });
  } catch (error) {
    console.error('Gemini API error:', error.message || error);

    // Extract error message for frontend
    let errorMessage = 'API_ERROR';
    if (error.message) {
      errorMessage = error.message;
    } else if (typeof error === 'string') {
      errorMessage = error;
    }

    // Return fallback response on error with error details
    return res.json({
      prompt,
      brand,
      mentioned: false,
      position: 0,
      raw: 'API_ERROR',
      error: errorMessage,
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Model: ${MODEL_NAME}, Temperature: ${TEMPERATURE}`);
  console.log(`API Key: ${process.env.GEMINI_API_KEY ? 'Configured' : 'Not configured'}`);
});

