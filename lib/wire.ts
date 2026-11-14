// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. The entry
// answers from a provider on the openai, anthropic or gemini wire, so every client wire times
// every upstream wire has a translation here:
//
//   - OpenAI client: its body goes to the provider as it stands (with the routed model's id);
//     an Anthropic provider gets the Chat Completions request rewritten to Messages (system
//     prompt, image blocks, tool definitions, tool results and tool choice included); a Gemini
//     provider gets it rewritten to generateContent (systemInstruction, contents, function
//     declarations, generationConfig).
//   - Anthropic client: the reverse — rewritten to Chat Completions for an openai-wire provider
//     or to generateContent for a gemini-wire provider, and passed through for an anthropic one.
//
// One simplification makes every combination exact: the upstream call is always the provider's
// non-streaming endpoint, whatever the client asked for. The simulated providers answer a
// stream and a plain call with the same text, the same usage and the same stop reason — only
// the framing differs — so the entry renders the client's JSON or SSE from the one parsed
// answer. The scorer records no streaming flag on a provider call, and the client's checks
// read only the response the entry sends. A response is built from the parsed answer alone, so
// a replay sees the same bytes.
//
// The provider's answer text is an echo of what it parsed (system, user text, tools, stop
// sequences, max_tokens, images, JSON mode), and the client compares that echo with what it
// sent. Every translation below passes those fields through untouched: a field rewritten or
// dropped loses the request. Two wire limits matter for routing (lib/plan.ts enforces them):
// an OpenAI JSON-mode request carries response_format, which only the openai wire (as
// response_format) and the gemini wire (as responseMimeType) can carry — an anthropic-wire
// provider echoes it absent. And usage must come back exactly as the provider reported it:
// OpenAI's prompt/completion tokens, Anthropic's input/output tokens, Gemini's usageMetadata.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The model listings use
// fixed timestamps for the same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json, type Wire } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** One provider answer, parsed out of its wire: the text, the usage, and the stop reason. */
export interface ParsedAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  const seen = new Set<string>();
  const data = models.flatMap((m) => {
    if (seen.has(m.id)) return [];
    seen.add(m.id);
    return [{ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider }];
  });
  return { object: 'list', data };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  const seen = new Set<string>();
  const data = models.flatMap((m) => {
    if (seen.has(m.id)) return [];
    seen.add(m.id);
    return [{ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT }];
  });
  return { data, has_more: false };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A provider's stop reason as the client's wire names its family: stop/end_turn/STOP are one family. */
function clientStop(providerStop: string, clientWire: 'openai' | 'anthropic'): string {
  if (providerStop === 'length' || providerStop === 'max_tokens' || providerStop === 'MAX_TOKENS') return clientWire === 'openai' ? 'length' : 'max_tokens';
  if (providerStop === 'tool_calls' || providerStop === 'function_call' || providerStop === 'tool_use') return clientWire === 'openai' ? 'tool_calls' : 'tool_use';
  if (providerStop === 'content_filter' || providerStop === 'refusal' || providerStop === 'SAFETY') return clientWire === 'openai' ? 'content_filter' : 'refusal';
  return clientWire === 'openai' ? 'stop' : 'end_turn';
}

/** Split a text into pieces the way the simulated providers stream it: 13 characters at a time. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Client request -> upstream request
// ---------------------------------------------------------------------------

function anthropicToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', input_schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

function geminiToolsOf(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length === 0 ? [] : [{ functionDeclarations: declarations }];
}

function maxTokensOf(body: Json): number | undefined {
  if (typeof body.max_tokens === 'number') return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') return body.max_completion_tokens;
  return undefined;
}

function stopStringsOf(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/** An Anthropic Messages body's tools as OpenAI function tools. */
function anthropicToolsToOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
  });
}

/** An OpenAI Chat Completions body's tools as Anthropic tools. */
function openAIToolsToAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const system = (Array.isArray(body.messages) ? body.messages : []).filter(isRecord).filter((m) => m.role === 'system');
  const systemText = system.map((m) => textOf(m.content)).filter((t) => t !== '').join('\n');
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({ role: 'function', parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }] });
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(part.image_url.url);
          if (data !== null) parts.push({ inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
          else parts.push({ fileData: { fileUri: part.image_url.url } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopStringsOf(body);
  if (stops.length > 0) config.stopSequences = stops;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsOf(body);
  if (tools.length > 0) out.tools = tools;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p and stop sequences pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopStringsOf(body);
  if (stops.length > 0) out.stop = stops;
  const tools = anthropicToolsToOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: '', response: { content: toolResultText(block.content) } } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const config: Json = {};
  const maxTokens = maxTokensOf(body);
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stopStringsOf(body);
  if (stops.length > 0) config.stopSequences = stops;
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = anthropicToolsOf(body);
  if (tools.length > 0) {
    out.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.input_schema })) }];
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request that names `model`. */
function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
        }
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }] });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = blockOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  const maxTokens = maxTokensOf(body);
  out.max_tokens = maxTokens !== undefined ? maxTokens : 1024;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopStringsOf(body);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  return out;
}

/**
 * The upstream request for one client body: the path under the provider's base, the headers, and
 * the JSON body. The call is always the provider's non-streaming endpoint: a streamed answer
 * carries the same text, usage and stop reason as the plain one, and the entry renders the
 * client's stream from the one parsed answer.
 */
export function upstreamRequest(clientApi: 'openai' | 'anthropic', body: Json, wire: Wire, model: string): { path: string; headers: Record<string, string>; body: Json } {
  if (wire === 'openai') {
    const upstream = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    delete upstream.stream;
    delete upstream.stream_options;
    return { path: '/v1/chat/completions', headers: { 'content-type': 'application/json', accept: 'application/json' }, body: upstream };
  }
  if (wire === 'anthropic') {
    const upstream = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    delete upstream.stream;
    return { path: '/v1/messages', headers: { 'content-type': 'application/json', accept: 'application/json', 'anthropic-version': '2023-06-01' }, body: upstream };
  }
  const upstream = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return { path: `/v1beta/models/${model}:generateContent`, headers: { 'content-type': 'application/json', accept: 'application/json' }, body: upstream };
}

// ---------------------------------------------------------------------------
// Upstream answer -> one parsed answer
// ---------------------------------------------------------------------------

/** A Chat Completions answer as one parsed answer; null when it is not one. */
function parseOpenAIAnswer(text: string): ParsedAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const answerText = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: answerText,
    usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages answer as one parsed answer; null when it is not one. */
function parseAnthropicAnswer(text: string): ParsedAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value) || value.type !== 'message' || !Array.isArray(value.content)) return null;
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: textOf(value.content),
    usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
    stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent answer as one parsed answer; null when it is not one. */
function parseGeminiAnswer(text: string): ParsedAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
  const candidate = isRecord(value.candidates[0]) ? value.candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const answerText = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: answerText,
    usage: { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) },
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

/** The provider's answer body as one parsed answer, in its wire's shape; null when it is not one. */
export function parseUpstreamAnswer(wire: Wire, text: string): ParsedAnswer | null {
  if (wire === 'openai') return parseOpenAIAnswer(text);
  if (wire === 'anthropic') return parseAnthropicAnswer(text);
  return parseGeminiAnswer(text);
}

// ---------------------------------------------------------------------------
// One parsed answer -> the client's response
// ---------------------------------------------------------------------------

/** A parsed answer as an OpenAI Chat Completions object, or as its SSE stream when asked. */
export function renderOpenAI(answer: ParsedAnswer, model: string, stream: boolean): string {
  const stop = clientStop(answer.stop, 'openai');
  const usage = { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output };
  if (!stream) {
    return JSON.stringify({
      id: 'chatcmpl-aggregate',
      object: 'chat.completion',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stop, logprobs: null }],
      usage,
    });
  }
  const frames: string[] = [];
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }));
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: stop }] }));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A parsed answer as an Anthropic Messages object, or as its SSE event sequence when asked. */
export function renderAnthropic(answer: ParsedAnswer, model: string, stream: boolean): string {
  const stop = clientStop(answer.stop, 'anthropic');
  if (!stream) {
    return JSON.stringify({
      id: 'msg_aggregate',
      type: 'message',
      role: 'assistant',
      model,
      content: [{ type: 'text', text: answer.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
    });
  }
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.usage.output } }));
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
