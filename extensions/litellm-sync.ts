/**
 * LiteLLM Model Sync Extension for Pi
 *
 * Keeps the Pi model selector in sync with LiteLLM's live model list.
 * On startup (and on /reload), fetches GET /v1/models from LiteLLM and
 * calls pi.registerProvider("litellm", ...) with the exact model IDs
 * that LiteLLM exposes — so the selector shows "bedrock/claude-sonnet-4.6"
 * rather than stale IDs baked into models.json.
 *
 * Configuration:
 *   Set LITELLM_BASE_URL env var or configure in models.json provider config
 *   Set LITELLM_API_KEY env var or configure in models.json provider config
 *
 * Priority: env vars > models.json provider config > defaults
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LiteLLMModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  /** Context window advertised by LiteLLM. Absent for some upstream providers. */
  max_input_tokens?: number | null;
  /** Max output tokens advertised by LiteLLM. Absent for some upstream providers. */
  max_output_tokens?: number | null;
}

interface LiteLLMModelsResponse {
  data: LiteLLMModel[];
  object: string;
}

/** Shape of GET /model/info (LiteLLM admin endpoint, also used by litellm-cost.ts). */
interface LiteLLMModelInfoEntry {
  model_name: string;
  model_info?: {
    max_input_tokens?: number | null;
    max_output_tokens?: number | null;
  } | null;
}

interface LiteLLMModelInfoResponse {
  data: LiteLLMModelInfoEntry[];
}

/** Per-model manual overrides, highest priority. */
interface ModelOverride {
  contextWindow?: number;
  maxTokens?: number;
  reasoning?: boolean;
}

interface ProviderConfig {
  baseUrl: string;
  apiKey: string;
  api: string;
  modelOverrides: Record<string, ModelOverride>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Derive display name from model id, e.g. "bedrock/claude-sonnet-4.6" → "Claude Sonnet 4.6 (Bedrock)" */
function displayName(id: string): string {
  const [prefix, ...rest] = id.split("/");
  const modelPart = rest.join("/");
  if (!modelPart) return id;

  const prefixLabel: Record<string, string> = {
    bedrock: "Bedrock",
    openrouter: "OpenRouter",
    openai: "OpenAI",
    azure: "Azure",
    vertex_ai: "Vertex AI",
    anthropic: "Anthropic",
    ollama: "Ollama",
  };

  const label = prefixLabel[prefix] ?? prefix.charAt(0).toUpperCase() + prefix.slice(1);
  const human = modelPart
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return `${human} (${label})`;
}

/** Guess capabilities from model id */
function modelMeta(id: string): {
  reasoning: boolean;
  contextWindow: number;
  maxTokens: number;
  thinkingLevelMap?: Record<string, string | null>;
} {
  const lower = id.toLowerCase();

  const isOpus = lower.includes("opus");
  const isSonnet = lower.includes("sonnet");
  const isHaiku = lower.includes("haiku");
  const isClaude = lower.includes("claude");
  const isClaude4x = /claude[\-.]?(opus|sonnet|haiku)?[\-.]?4/.test(lower);
  const isClaude3x = /claude[\-.]?3[\-.]?(5|7)/.test(lower);
  const isModernClaude = isClaude4x || isClaude3x;

  // Reasoning-capable families beyond Claude. Without this, thinking models from
  // other vendors were silently registered as non-reasoning.
  const isOtherReasoning =
    /thinking|reasoner|deepseek-(r|v[4-9])|glm-[5-9]|qwen[3-9]|nemotron|minimax|cogito|kimi-k[3-9]/.test(
      lower
    );

  const reasoning = isModernClaude || isOtherReasoning;

  // Heuristic fallback ONLY. Real values come from LiteLLM metadata when present.
  const contextWindow = isModernClaude ? 200000 : isClaude ? 200000 : 128000;

  const maxTokens = isOpus
    ? 32768
    : isSonnet
    ? 16384
    : isHaiku
    ? 8192
    : 16384;

  const thinkingLevelMap: Record<string, string | null> | undefined = reasoning
    ? isOpus
      ? { low: "low", medium: "medium", high: "high", xhigh: "max" }
      : { low: "low", medium: "medium", high: "high", xhigh: null }
    : undefined;

  return { reasoning, contextWindow, maxTokens, thinkingLevelMap };
}

// ---------------------------------------------------------------------------
// Config resolution
// ---------------------------------------------------------------------------

function resolveConfig(): ProviderConfig {
  const defaults: ProviderConfig = {
    baseUrl: "http://localhost:4000",
    apiKey: "sk-cedar-local",
    api: "anthropic-messages",
    modelOverrides: {},
  };

  // Try reading from models.json for non-env-var config
  let fromModelsJson: Partial<ProviderConfig> = {};
  const modelsJsonPath = join(homedir(), ".pi", "agent", "models.json");
  if (existsSync(modelsJsonPath)) {
    try {
      const raw = readFileSync(modelsJsonPath, "utf-8");
      // Strip // comments before parsing
      const stripped = raw
        .replace(/"(?:\\.|[^"\\])*"|\/\/[^\n]*/g, (m) => (m[0] === '"' ? m : ""))
        .replace(/"(?:\\.|[^"\\])*"|,(\s*[}\]])/g, (m, tail) => tail ?? (m[0] === '"' ? m : ""));
      const parsed = JSON.parse(stripped) as { providers?: Record<string, any> };
      const p = parsed?.providers?.litellm ?? {};
      fromModelsJson = {
        baseUrl: p.baseUrl,
        apiKey: p.apiKey,
        api: p.api,
        modelOverrides: p.modelOverrides,
      };
    } catch {
      // Ignore parse errors — fall through to defaults
    }
  }

  // Priority: env vars > models.json > defaults
  return {
    baseUrl: process.env.LITELLM_BASE_URL ?? fromModelsJson.baseUrl ?? defaults.baseUrl,
    apiKey: process.env.LITELLM_API_KEY ?? fromModelsJson.apiKey ?? defaults.apiKey,
    api: fromModelsJson.api ?? defaults.api,
    modelOverrides: fromModelsJson.modelOverrides ?? defaults.modelOverrides,
  };
}

// ---------------------------------------------------------------------------
// Secondary metadata source
// ---------------------------------------------------------------------------

/**
 * Fetch GET /model/info for models whose /v1/models entry omits token limits.
 * Best-effort: this endpoint may be restricted, so failures degrade silently
 * back to the /v1/models values and then the heuristic.
 */
async function fetchModelInfo(
  config: ProviderConfig
): Promise<Map<string, { contextWindow?: number; maxTokens?: number }>> {
  const out = new Map<string, { contextWindow?: number; maxTokens?: number }>();
  try {
    const res = await fetch(`${config.baseUrl}/model/info`, {
      headers: { Authorization: `Bearer ${config.apiKey}` },
    });
    if (!res.ok) return out;

    const payload = (await res.json()) as LiteLLMModelInfoResponse;
    for (const entry of payload.data ?? []) {
      const info = entry.model_info;
      if (!entry.model_name || !info) continue;
      out.set(entry.model_name, {
        contextWindow: info.max_input_tokens ?? undefined,
        maxTokens: info.max_output_tokens ?? undefined,
      });
    }
  } catch {
    // Ignore — /v1/models plus the heuristic still produce a usable registration.
  }
  return out;
}

// ---------------------------------------------------------------------------
// Core sync function
// ---------------------------------------------------------------------------

async function syncModels(pi: ExtensionAPI): Promise<void> {
  const config = resolveConfig();

  let models: LiteLLMModel[];
  try {
    const res = await fetch(`${config.baseUrl}/v1/models`, {
      headers: { Authorization: `Bearer ${config.apiKey}` },
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }
    const payload = (await res.json()) as LiteLLMModelsResponse;
    models = payload.data ?? [];
  } catch (err) {
    console.error(
      `[litellm-sync] Failed to fetch models from ${config.baseUrl}: ${err instanceof Error ? err.message : err}`
    );
    return;
  }

  if (models.length === 0) {
    console.warn("[litellm-sync] No models returned from LiteLLM — skipping registration.");
    return;
  }

  // Only query /model/info when /v1/models left gaps.
  const needsModelInfo = models.some((m) => !m.max_input_tokens || !m.max_output_tokens);
  const modelInfo = needsModelInfo
    ? await fetchModelInfo(config)
    : new Map<string, { contextWindow?: number; maxTokens?: number }>();

  let fromProvider = 0;
  let fromHeuristic = 0;
  const unresolved: string[] = [];

  const registered = models.map((m) => {
    const meta = modelMeta(m.id);
    const info = modelInfo.get(m.id);
    const override = config.modelOverrides[m.id] ?? {};

    // Priority: explicit override > /v1/models > /model/info > heuristic.
    const contextWindow =
      override.contextWindow ?? m.max_input_tokens ?? info?.contextWindow ?? meta.contextWindow;
    const maxTokens =
      override.maxTokens ?? m.max_output_tokens ?? info?.maxTokens ?? meta.maxTokens;

    const contextFromProvider =
      override.contextWindow != null || m.max_input_tokens != null || info?.contextWindow != null;
    if (contextFromProvider) fromProvider++;
    else {
      fromHeuristic++;
      unresolved.push(m.id);
    }

    return {
      id: m.id,
      name: displayName(m.id),
      reasoning: override.reasoning ?? meta.reasoning,
      thinkingLevelMap: meta.thinkingLevelMap,
      input: ["text", "image"] as ["text", "image"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow,
      maxTokens,
    };
  });

  pi.registerProvider("litellm", {
    baseUrl: config.baseUrl,
    apiKey: config.apiKey,
    api: config.api as any,
    models: registered,
  });

  console.log(
    `[litellm-sync] Registered ${models.length} model(s); ` +
      `${fromProvider} context window(s) from LiteLLM metadata, ${fromHeuristic} from heuristic fallback.`
  );
  if (unresolved.length > 0) {
    console.warn(
      `[litellm-sync] No token limits advertised for: ${unresolved.join(", ")}. ` +
        `Using heuristic defaults — set providers.litellm.modelOverrides in models.json to correct them.`
    );
  }
}

// ---------------------------------------------------------------------------
// Extension entry point (async factory — pi awaits before session_start)
// ---------------------------------------------------------------------------

export default async function (pi: ExtensionAPI) {
  await syncModels(pi);

  // Re-sync on /reload so the selector stays fresh without restarting pi
  pi.on("session_start", async (event) => {
    if (event.reason === "reload") {
      await syncModels(pi);
    }
  });
}
