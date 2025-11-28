const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const stringSimilarity = require('string-similarity');
const Sentiment = require('sentiment');

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
  : ['https://gemini-brand-mention-checker-omega.vercel.app/', 'http://localhost:5173'];

// app.use(
//   cors({
//     origin: (origin, callback) => {
//       if (!origin || allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
//         callback(null, true);
//       } else {
//         callback(new Error('Not allowed by CORS'));
//       }
//     },
//     credentials: true,
//   })
// );
app.use(cors());
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
 * Position is calculated considering the prompt context for better accuracy
 * For recommendation/list prompts, position reflects the rank/order in the list
 */
function findBrandMention(text, brand, prompt = '') {
  if (!text || !brand) {
    return { mentioned: false, position: 0 };
  }

  const brandLower = brand.toLowerCase().trim();
  const textLower = text.toLowerCase();
  const promptLower = prompt.toLowerCase();
  const brandTokens = brandLower.split(/\s+/).filter(token => token.length > 0);
  
  // Extract all words from text for position calculation
  const allWords = textLower.match(/\b\w+\b/g) || [];
  
  let earliestWordPosition = Infinity;
  let listRank = null;

  // Check if prompt is asking for recommendations/list (common patterns)
  const isRecommendationPrompt = /recommend|list|best|top|suggest|compare|options/i.test(promptLower);
  
  // Try to find numbered list patterns (1., 2., 3., etc.)
  // Pattern matches: "1. **Brand Name:**" or "1. Brand Name:" etc.
  // More flexible pattern that handles various formats
  const listPatterns = [
    // Pattern 1: "1.  **Brand Name:**" with markdown
    /^\s*(\d+)\.\s+(?:\*{1,3})?\s*([^\n*:]+?)(?:\*{1,3})?\s*:?\s*(?:\n|$)/gm,
    // Pattern 2: "1. Brand Name:" without markdown
    /^\s*(\d+)\.\s+([^\n:]+?)\s*:?\s*(?:\n|$)/gm,
    // Pattern 3: "1. Brand Name" without colon
    /^\s*(\d+)\.\s+(?:\*{0,3})?\s*([^\n]+?)(?:\*{0,3})?\s*(?:\n\n|$)/gm
  ];
  
  const listMatches = [];
  
  for (const pattern of listPatterns) {
    pattern.lastIndex = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const number = parseInt(match[1]);
      // Clean up the content - remove markdown formatting but keep the text
      let content = match[2].replace(/\*+/g, '').toLowerCase().trim();
      // Remove trailing colon if present
      content = content.replace(/:\s*$/, '').trim();
      
      // Check if we already have this list item at this position (avoid duplicates)
      const exists = listMatches.some(item => 
        item.number === number && Math.abs(item.index - match.index) < 10
      );
      
      if (!exists && content.length > 0) {
        const fullMatch = match[0].toLowerCase();
        listMatches.push({ number, content, fullMatch, index: match.index });
      }
    }
  }
  
  // Sort list matches by index to ensure correct order
  listMatches.sort((a, b) => a.index - b.index);
  
  // Remove duplicates (keep the first occurrence of each number)
  const seenNumbers = new Set();
  const uniqueListMatches = [];
  for (const item of listMatches) {
    if (!seenNumbers.has(item.number)) {
      seenNumbers.add(item.number);
      uniqueListMatches.push(item);
    }
  }

  // PRIORITY STRATEGY: For recommendation prompts, check list items FIRST
  // This ensures we get the correct list rank (1, 2, 3) instead of word position
  if (isRecommendationPrompt && uniqueListMatches.length > 0) {
    for (const listItem of uniqueListMatches) {
      // Get the full text of the list item (including the number and formatting)
      // Extract text from the original text at the list item position
      const itemStartIndex = listItem.index;
      // Find where this list item ends (next list item or end of text)
      let itemEndIndex = text.length;
      for (let k = 0; k < uniqueListMatches.length; k++) {
        if (uniqueListMatches[k].index > itemStartIndex) {
          itemEndIndex = uniqueListMatches[k].index;
          break;
        }
      }
      // Get the full list item text (first 500 chars to avoid getting too much)
      const fullItemText = text.substring(itemStartIndex, Math.min(itemStartIndex + 500, itemEndIndex)).toLowerCase();
      const itemContentLower = listItem.content.toLowerCase();
      
      // For multi-word brands, check if all tokens appear in the list item
      if (brandTokens.length > 1) {
        // Check if all brand tokens are present in the list item content or full text
        const allTokensInContent = brandTokens.every(token => 
          itemContentLower.includes(token.toLowerCase())
        );
        const allTokensInFull = brandTokens.every(token => 
          fullItemText.includes(token.toLowerCase())
        );
        
        // Also check for exact phrase match
        const hasExactPhrase = itemContentLower.includes(brandLower) || fullItemText.includes(brandLower);
        
        if (allTokensInContent || allTokensInFull || hasExactPhrase) {
          listRank = listItem.number;
          // Also set word position for fallback
          if (earliestWordPosition === Infinity) {
            // Count words from start of text to start of this list item
            const textBeforeItem = textLower.substring(0, itemStartIndex);
            const wordsBeforeItem = textBeforeItem.match(/\b\w+\b/g) || [];
            // Find brand position within the list item
            const brandIndexInItem = fullItemText.indexOf(brandLower);
            if (brandIndexInItem !== -1) {
              const itemTextBeforeBrand = fullItemText.substring(0, brandIndexInItem);
              const wordsInItemBeforeBrand = itemTextBeforeBrand.match(/\b\w+\b/g) || [];
              earliestWordPosition = wordsBeforeItem.length + wordsInItemBeforeBrand.length + 1;
            } else {
              // If exact phrase not found, just use the start of the list item
              earliestWordPosition = wordsBeforeItem.length + 1;
            }
          }
          break; // Found it, stop searching
        }
      } else {
        // Single-word brand
        if (itemContentLower.includes(brandLower) || fullItemText.includes(brandLower) || 
            fuzzyMatch(itemContentLower, brandLower)) {
          listRank = listItem.number;
          if (earliestWordPosition === Infinity) {
            const textBeforeItem = textLower.substring(0, itemStartIndex);
            const wordsBeforeItem = textBeforeItem.match(/\b\w+\b/g) || [];
            earliestWordPosition = wordsBeforeItem.length + 1;
          }
          break;
        }
      }
    }
  }

  // Strategy 1: For multi-word brands, find the complete phrase as sequential words
  // Only do this if we haven't found a list rank yet
  if (brandTokens.length > 1 && listRank === null) {
    // Try to find the brand phrase as sequential words
    for (let i = 0; i <= allWords.length - brandTokens.length; i++) {
      let allMatch = true;
      for (let j = 0; j < brandTokens.length; j++) {
        if (i + j >= allWords.length || !fuzzyMatch(allWords[i + j], brandTokens[j])) {
          allMatch = false;
          break;
        }
      }
      if (allMatch) {
        earliestWordPosition = i + 1;
        break;
      }
    }
    
    // Fallback: exact phrase match
    if (earliestWordPosition === Infinity) {
      const exactPhraseIndex = textLower.indexOf(brandLower);
      if (exactPhraseIndex !== -1) {
        const textBefore = textLower.substring(0, exactPhraseIndex);
        const wordsBefore = textBefore.match(/\b\w+\b/g) || [];
        earliestWordPosition = wordsBefore.length + 1;
      }
    }
  } else if (brandTokens.length === 1 && listRank === null) {
    // Strategy 2: For single-word brands
    for (let i = 0; i < allWords.length; i++) {
      const word = allWords[i];
      if (fuzzyMatch(word, brandLower)) {
        earliestWordPosition = i + 1;
        break;
      }
    }
  }

  // Strategy 3: Final fallback
  if (earliestWordPosition === Infinity && textLower.includes(brandLower)) {
    const index = textLower.indexOf(brandLower);
    const textBefore = textLower.substring(0, index);
    const wordsBefore = textBefore.match(/\b\w+\b/g) || [];
    earliestWordPosition = wordsBefore.length + 1;
  }

  // For recommendation prompts, prefer list rank over word position if available
  // This gives more meaningful position (1st, 2nd, 3rd recommendation)
  const finalPosition = (isRecommendationPrompt && listRank !== null) 
    ? listRank 
    : (earliestWordPosition !== Infinity ? earliestWordPosition : 0);

  return earliestWordPosition !== Infinity || listRank !== null
    ? { mentioned: true, position: finalPosition }
    : { mentioned: false, position: 0 };
}

/**
 * Analyze sentiment of brand mentions in the response text
 * Extracts context around brand mentions and analyzes sentiment
 */
function analyzeBrandSentiment(text, brand) {
  if (!text || !brand) {
    return {
      sentiment: 'neutral',
      score: 0,
      confidence: 0,
      context: []
    };
  }

  const sentiment = new Sentiment();
  const brandLower = brand.toLowerCase().trim();
  const textLower = text.toLowerCase();
  const brandTokens = brandLower.split(/\s+/).filter(token => token.length > 0);
  
  // Find all sentences or contexts that mention the brand
  const contexts = [];
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  
  // Also split by newlines and list items for better context extraction
  const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 0);
  
  // Find contexts containing the brand
  for (const sentence of sentences) {
    const sentenceLower = sentence.toLowerCase();
    // Check if sentence contains brand (exact or fuzzy match)
    if (sentenceLower.includes(brandLower) || 
        brandTokens.some(token => sentenceLower.includes(token))) {
      contexts.push(sentence.trim());
    }
  }
  
  // Also check paragraphs for broader context
  for (const paragraph of paragraphs) {
    const paraLower = paragraph.toLowerCase();
    if (paraLower.includes(brandLower) || 
        brandTokens.some(token => paraLower.includes(token))) {
      // Extract a relevant snippet (around 200 chars around brand mention)
      const brandIndex = paraLower.indexOf(brandLower);
      if (brandIndex !== -1) {
        const start = Math.max(0, brandIndex - 100);
        const end = Math.min(paragraph.length, brandIndex + brandLower.length + 100);
        const snippet = paragraph.substring(start, end).trim();
        if (snippet && !contexts.includes(snippet)) {
          contexts.push(snippet);
        }
      }
    }
  }
  
  // If no specific contexts found, analyze the entire text
  if (contexts.length === 0) {
    contexts.push(text);
  }
  
  // Analyze sentiment for each context
  const sentimentResults = contexts.map(context => {
    const result = sentiment.analyze(context);
    return {
      text: context.substring(0, 200) + (context.length > 200 ? '...' : ''),
      score: result.score,
      comparative: result.comparative,
      tokens: result.tokens,
      words: result.words,
      positive: result.positive,
      negative: result.negative
    };
  });
  
  // Calculate overall sentiment
  const totalScore = sentimentResults.reduce((sum, r) => sum + r.score, 0);
  const avgScore = sentimentResults.length > 0 ? totalScore / sentimentResults.length : 0;
  const avgComparative = sentimentResults.reduce((sum, r) => sum + r.comparative, 0) / Math.max(1, sentimentResults.length);
  
  // Determine sentiment label
  let sentimentLabel = 'neutral';
  let confidence = Math.abs(avgComparative);
  
  if (avgScore > 2) {
    sentimentLabel = 'very_positive';
  } else if (avgScore > 0.5) {
    sentimentLabel = 'positive';
  } else if (avgScore < -2) {
    sentimentLabel = 'very_negative';
  } else if (avgScore < -0.5) {
    sentimentLabel = 'negative';
  } else {
    sentimentLabel = 'neutral';
  }
  
  return {
    sentiment: sentimentLabel,
    score: Math.round(avgScore * 100) / 100, // Round to 2 decimal places
    comparative: Math.round(avgComparative * 100) / 100,
    confidence: Math.round(confidence * 100) / 100,
    contextCount: contexts.length,
    contexts: sentimentResults.slice(0, 5) // Return top 5 contexts
  };
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
      sentiment: 'neutral',
      sentimentScore: 0,
      sentimentConfidence: 0,
      sentimentContexts: [],
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

    // Pass prompt context to help determine position more accurately
    const { mentioned, position } = findBrandMention(responseText, brand, prompt);

    // Analyze sentiment of brand mentions
    const sentimentAnalysis = analyzeBrandSentiment(responseText, brand);

    return res.json({
      prompt,
      brand,
      mentioned,
      position,
      sentiment: sentimentAnalysis.sentiment,
      sentimentScore: sentimentAnalysis.score,
      sentimentConfidence: sentimentAnalysis.confidence,
      sentimentContexts: sentimentAnalysis.contexts,
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
      sentiment: 'neutral',
      sentimentScore: 0,
      sentimentConfidence: 0,
      sentimentContexts: [],
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

