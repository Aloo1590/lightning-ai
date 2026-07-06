require("dotenv").config();
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

// Force thinking mode on for every request by default. Set FORCE_THINKING=false
// in your environment to disable this and only think when the client asks for it.
const FORCE_THINKING = process.env.FORCE_THINKING !== "false";

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
  const { messages, enable_reasoning, chat_template_kwargs, ...rest } = body;

  const final = {
    ...rest,
    model,
    messages: normalizeMessages(messages),
  };

  // Force thinking mode on by default (FORCE_THINKING env var), unless the
  // client explicitly sent enable_reasoning: false to opt out for this
  // specific request.
  const wantsThinking = enable_reasoning !== false && (FORCE_THINKING || enable_reasoning === true);

  if (wantsThinking) {
    final.chat_template_kwargs = {
      ...(chat_template_kwargs || {}),
      enable_thinking: true,
    };
  } else if (chat_template_kwargs) {
    final.chat_template_kwargs = chat_template_kwargs;
  }

  return final;
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
    let thinkOpen = false;

    const transformLine = (line) => {
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
        const delta = { ...(choice.delta || {}) };
        const reasoning = delta.reasoning_content;
        let content = flattenContent(delta.content) || "";

        if (reasoning) {
          content = (thinkOpen ? "" : "<think>") + reasoning + content;
          thinkOpen = true;
        } else if (thinkOpen) {
          content = "</think>" + content;
          thinkOpen = false;
        }

        delta.content = content;
        delete delta.reasoning_content;

        return { ...choice, delta };
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

        res.write(lines.map(transformLine).join("\n") + (lines.length ? "\n" : ""));
      }

      let tail = "";
      if (buffer) tail += transformLine(buffer);
      if (thinkOpen) {
        tail += (tail ? "\n" : "") + `data: ${JSON.stringify({
          choices: [{ index: 0, delta: { content: "</think>" } }],
        })}`;
      }
      if (tail) res.write(tail);
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
        if (!choice.message) return choice;

        const reasoning = choice.message.reasoning_content;
        const content = flattenContent(choice.message.content) || "";

        choice.message.content = reasoning
          ? `<think>\n${reasoning}\n</think>\n\n${content}`
          : content;
        delete choice.message.reasoning_content;

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
