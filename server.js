const express = require("express");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.options("*", cors()); // answer CORS preflight explicitly for every route
app.use(express.json({ limit: "20mb" }));

const LIGHTNING_API_KEY = process.env.LIGHTNING_API_KEY;
const LIGHTNING_API_BASE = "https://lightning.ai/api/v1";
const REQUEST_TIMEOUT_MS = 120_000;

if (!LIGHTNING_API_KEY) {
  console.warn("⚠️ LIGHTNING_API_KEY not set");
}

/* ------------------ HELPERS ------------------ */

function resolveModel(model) {
  if (typeof model !== "string" || model.trim().length === 0) {
    throw new Error("model is required");
  }
  return model.trim();
}

/**
 * Lightning AI expects message content as an array of content parts:
 *   [{ "type": "text", "text": "..." }]
 * Janitor AI (like most OpenAI-compatible clients) sends plain strings:
 *   "content": "Hello, world!"
 * This converts string content into the part-array format Lightning wants,
 * without touching content that's already in array form (e.g. images).
 */
function normalizeMessages(messages) {
  return messages.map((msg) => {
    if (typeof msg.content === "string") {
      return { ...msg, content: [{ type: "text", text: msg.content }] };
    }
    return msg;
  });
}

/**
 * Some OpenAI-compatible clients (and Janitor's renderer) expect
 * `message.content` back as a plain string. If Lightning ever returns the
 * part-array form, flatten it back to a string for compatibility.
 */
function flattenContent(content) {
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : part.text || ""))
      .join("");
  }
  return content;
}

function buildBody(body, model) {
  const { messages, ...rest } = body;
  return {
    ...rest,
    model,
    messages: normalizeMessages(messages),
  };
}

/* ------------------ ROUTES ------------------ */

app.get("/health", (_, res) => {
  res.json({ ok: true });
});

app.get("/v1/models", async (_, res) => {
  if (!LIGHTNING_API_KEY) {
    return res.status(401).json({ error: "Missing LIGHTNING_API_KEY" });
  }

  try {
    const upstream = await fetch(`${LIGHTNING_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${LIGHTNING_API_KEY}` },
    });
    const data = await upstream.json().catch(() => ({}));
    res.status(upstream.status).json(data);
  } catch (err) {
    console.error("models fetch failed:", err);
    res.status(502).json({ error: "failed to fetch model list from Lightning" });
  }
});

/* ------------------ MAIN ------------------ */

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body;

  if (!LIGHTNING_API_KEY) {
    return res.status(401).json({ error: "Missing LIGHTNING_API_KEY" });
  }

  if (!body || !Array.isArray(body.messages)) {
    return res.status(400).json({ error: "messages required" });
  }

  let model;
  try {
    model = resolveModel(body.model);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }

  const lightningBody = buildBody(body, model);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  res.on("close", () => controller.abort());

  let upstream;
  try {
    upstream = await fetch(`${LIGHTNING_API_BASE}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LIGHTNING_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(lightningBody),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError") {
      return res.headersSent ? res.end() : res.status(504).json({ error: "timeout" });
    }
    console.error(err);
    return res.status(502).json({ error: "upstream request failed" });
  }

  /* -------- STREAM -------- */
  if (body.stream) {
    if (!upstream.ok) {
      clearTimeout(timeout);
      const err = await upstream.json().catch(() => ({}));
      return res.status(upstream.status).json(err);
    }

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    const reader = upstream.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const flattenLine = (line) => {
      if (!line.startsWith("data: ")) return line;
      const jsonStr = line.slice(6).trim();
      if (!jsonStr || jsonStr === "[DONE]") return line;

      let parsed;
      try {
        parsed = JSON.parse(jsonStr);
      } catch {
        return line;
      }
      if (!parsed.choices) return line;

      parsed.choices = parsed.choices.map((choice) => {
        if (choice.delta && "content" in choice.delta) {
          choice.delta.content = flattenContent(choice.delta.content);
        }
        return choice;
      });

      return `data: ${JSON.stringify(parsed)}`;
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep possibly-incomplete trailing line

        res.write(lines.map(flattenLine).join("\n") + (lines.length ? "\n" : ""));
      }
      if (buffer) res.write(flattenLine(buffer));
    } catch (err) {
      if (err.name !== "AbortError") console.error("stream error:", err);
    } finally {
      clearTimeout(timeout);
      reader.cancel().catch(() => {});
      res.end();
    }
    return;
  }

  /* -------- NON STREAM -------- */
  try {
    const data = await upstream.json();

    if (Array.isArray(data.choices)) {
      data.choices = data.choices.map((choice) => {
        if (choice.message && "content" in choice.message) {
          choice.message.content = flattenContent(choice.message.content);
        }
        return choice;
      });
    }

    res.status(upstream.status).json(data);
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: "invalid upstream response" });
  } finally {
    clearTimeout(timeout);
  }
});

/* ------------------ FALLBACK ------------------ */

app.all("*", (_, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`running on ${PORT}`);
});
