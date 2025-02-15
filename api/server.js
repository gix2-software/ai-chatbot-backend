import express from "express";
import pkg from "body-parser";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const { json } = pkg;
app.use(json());

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

// ✅ FIX: Ensure metadata is stored in Pinecone
app.post("/embed", async (req, res) => {
  console.log("Request received");
  console.log(req.body);

  const texts = req.body;

  // Ensure the input is an array of objects with a "text" field
  if (
    !Array.isArray(texts) ||
    !texts.every((item) => item.text && typeof item.text === "string")
  ) {
    return res.status(400).json({
      error:
        "Invalid input format, expected an array of objects with a 'text' field.",
    });
  }

  try {
    const embeddings = [];

    // Iterate over each text item and generate its embedding
    for (const { text } of texts) {
      const response = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: text,
      });

      console.log("Embedding response:", response);

      const embedding = response.data[0].embedding;

      // Store the embedding along with metadata
      await index.upsert([
        {
          id: `text-${Date.now()}`,
          values: embedding,
          metadata: { text }, // Store metadata for each text
        },
      ]);

      embeddings.push({ text, embedding }); // Collect embeddings for response
    }

    res
      .status(200)
      .send({ message: "Texts embedded successfully", embeddings });
  } catch (error) {
    console.error("Error in /embed:", error);
    res.status(500).send({ error: error.message });
  }
});

app.get("/", (req, res) => {
  res.send("Hello, World!");
});

// ✅ FIX: Ensure query retrieves stored context
app.post("/chat", async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== "string") {
    return res.status(400).json({ error: "Invalid query input" });
  }

  try {
    // Generate embedding for query
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: [query],
    });

    const embedding = embeddingResponse.data[0].embedding;

    // Query Pinecone for similar embeddings
    const pineconeResponse = await index.query({
      vector: embedding,
      topK: 5,
      includeValues: false,
      includeMetadata: true, // ✅ Ensure metadata is retrieved
    });

    console.log(
      "Pinecone Query Response:",
      JSON.stringify(pineconeResponse, null, 2)
    );

    if (!pineconeResponse.matches || pineconeResponse.matches.length === 0) {
      return res.status(200).json({ response: "No relevant context found." });
    }

    // ✅ FIX: Ensure metadata extraction is correct
    const context = pineconeResponse.matches
      .map((match) => match.metadata?.text) // Extract stored text
      .filter(Boolean) // Remove empty values
      .join("\n");

    if (!context) {
      return res.status(200).json({
        response: "I can only answer questions related to Gix2 Software.",
      });
    }

    // Create structured messages
    const messages = [
      {
        role: "system",
        content:
          "You are the AI chatbot of Gix2 Software. Only answer based on the provided company knowledge and context. If you don't know the answer, reply with: 'I can only answer questions related to Gix2 Software.'",
      },
      {
        role: "user",
        content: `Context: ${context}\n\nQuestion: ${query}`,
      },
    ];

    // Call OpenAI to generate a response
    const chatResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages,
      response_format: "json", // Forces the AI to return structured data
    });

    if (!chatResponse.choices || chatResponse.choices.length === 0) {
      return res
        .status(500)
        .json({ error: "OpenAI returned an empty response" });
    }

    // Return AI-generated response
    res.status(200).json({
      response: chatResponse.choices[0].message.content,
    });
  } catch (error) {
    console.error("Error in /chat:", error);
    res.status(500).json({
      error: error.message || "An error occurred while processing your request",
    });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
