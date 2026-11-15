// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

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

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
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
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/** --- Upstream wires the baseline did not speak: Anthropic and Gemini. --- */

/** One SSE frame read off an upstream stream: its event name (null when unnamed) and its data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split an upstream SSE stream into frames, in order. */
export async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): SseFrame[] => {
    const out: SseFrame[] = [];
    for (const raw of text.split('\n\n')) {
      if (raw.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
      }
      out.push({ event, data: data.join('\n') });
    }
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const chunk = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const frame of framesOf(chunk)) yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const frame of framesOf(buffer)) yield frame;
  } finally {
    reader.releaseLock();
  }
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const id = typeof call.id === 'string' ? call.id : '';
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (id !== '') toolNames.set(id, name);
        blocks.push({ type: 'tool_use', id: id !== '' ? id : `toolu_${blocks.length}`, name, input: argumentsOf(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const text = typeof content === 'string' ? content : textOf(content);
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: id, content: text }] });
      continue;
    }
    // A user message: text and image_url parts become text and image blocks.
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function openAIStopOf(anthropicStop: unknown): string {
  if (anthropicStop === 'max_tokens') return 'length';
  if (anthropicStop === 'tool_use') return 'tool_calls';
  if (anthropicStop === 'refusal') return 'content_filter';
  return 'stop';
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object naming `model`. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const message: Json = { role: 'assistant', content: texts.join('\n') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  return {
    id: `chatcmpl_${typeof body.id === 'string' ? body.id : 'sim'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStopOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** One OpenAI SSE chunk object, as a data frame. */
function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
}

/**
 * An Anthropic provider's event stream as OpenAI chunks: the opening role chunk, one content chunk per
 * text delta, the finish chunk from the provider's stop reason, a usage chunk when the client asked for
 * one, then [DONE]. `includeUsage` is the client's stream_options.include_usage.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string, includeUsage: boolean): AsyncGenerator<string> {
  let opened = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  const toolArgs = new Map<number, string>();
  const toolIds = new Map<number, string>();
  const toolNames = new Map<number, string>();
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      if (!opened) {
        opened = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
    } else if (frame.event === 'content_block_start') {
      if (!opened) {
        opened = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      const block = isRecord(data.content_block) ? data.content_block : {};
      const index = typeof data.index === 'number' ? data.index : 0;
      if (block.type === 'tool_use') {
        toolIds.set(index, typeof block.id === 'string' ? block.id : `call_${index}`);
        toolNames.set(index, typeof block.name === 'string' ? block.name : '');
        toolArgs.set(index, '');
      }
    } else if (frame.event === 'content_block_delta') {
      if (!opened) {
        opened = true;
        yield openAIChunk(model, { role: 'assistant', content: '' }, null);
      }
      const delta = isRecord(data.delta) ? data.delta : {};
      const index = typeof data.index === 'number' ? data.index : 0;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield openAIChunk(model, { content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        toolArgs.set(index, (toolArgs.get(index) ?? '') + delta.partial_json);
        yield openAIChunk(model, { tool_calls: [{ index, id: toolIds.get(index) ?? `call_${index}`, type: 'function', function: { name: toolNames.get(index) ?? '', arguments: delta.partial_json } }] }, null);
      }
    } else if (frame.event === 'message_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === 'string') stopReason = delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  if (!opened) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  yield openAIChunk(model, {}, openAIStopOf(stopReason));
  if (includeUsage) {
    yield sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
  }
  yield 'data: [DONE]\n\n';
}

/** --- Gemini upstream wire. --- */

/** A Gemini finishReason as an OpenAI finish_reason. */
export function openAIStopOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
export function anthropicStopOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** One image URL as a Gemini part: a data URL becomes inlineData, a plain URL becomes fileData. */
function geminiImagePart(url: string): Json {
  const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { fileUri: url } };
}

/** OpenAI tools as Gemini functionDeclarations. */
function geminiToolsOfOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** Anthropic tools as Gemini functionDeclarations. */
function geminiToolsOfAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : [];
}

/** An OpenAI Chat Completions request as a Gemini generateContent body (the model goes in the URL). */
export function openAIToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        const id = typeof call.id === 'string' ? call.id : '';
        const name = typeof call.function.name === 'string' ? call.function.name : '';
        if (id !== '') toolNames.set(id, name);
        parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      const text = typeof content === 'string' ? content : textOf(content);
      contents.push({ role: 'function', parts: [{ functionResponse: { name: toolNames.get(id) ?? '', response: { content: text } } }] });
      continue;
    }
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(geminiImagePart(part.image_url.url));
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiToolsOfOpenAI(body);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (choice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (choice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  }
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body (the model goes in the URL). */
export function anthropicToGeminiRequest(body: Json): Json {
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            const id = typeof block.id === 'string' ? block.id : '';
            const name = typeof block.name === 'string' ? block.name : '';
            if (id !== '') toolNames.set(id, name);
            parts.push({ functionCall: { name, args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof content === 'string' && content !== '') parts.push({ text: content });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message: text and image blocks become Gemini parts; tool results become function responses.
    if (typeof content === 'string') {
      contents.push({ role: 'user', parts: [{ text: content }] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    const responses: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        if (block.source.type === 'base64' && typeof block.source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
        } else if (block.source.type === 'url' && typeof block.source.url === 'string') parts.push(geminiImagePart(block.source.url));
      } else if (block.type === 'tool_result') {
        const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
        responses.push({ functionResponse: { name: toolNames.get(id) ?? '', response: { content: toolResultText(block.content) } } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
    if (responses.length > 0) contents.push({ role: 'function', parts: responses });
  }

  const out: Json = { contents };
  const systemText = textOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsOfAnthropic(body);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
    else if (choice.type === 'any') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  }
  return out;
}

/** The text and tool calls of a Gemini candidate's parts. */
function geminiPartsOf(body: Json): { text: string; calls: Json[]; finish: unknown; input: number; output: number } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  const texts: string[] = [];
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text: texts.join(''), calls, finish: candidate.finishReason, input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object naming `model`. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const parsed = geminiPartsOf(body);
  const message: Json = { role: 'assistant', content: parsed.text };
  if (parsed.calls.length > 0) message.tool_calls = parsed.calls;
  return {
    id: `chatcmpl_${typeof body.modelVersion === 'string' ? body.modelVersion : 'sim'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: parsed.calls.length > 0 ? 'tool_calls' : openAIStopOfGemini(parsed.finish), logprobs: null }],
    usage: { prompt_tokens: parsed.input, completion_tokens: parsed.output, total_tokens: parsed.input + parsed.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object naming `model`. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const parsed = geminiPartsOf(body);
  const content: Json[] = [{ type: 'text', text: parsed.text }];
  for (const call of parsed.calls) {
    if (!isRecord(call.function)) continue;
    content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
  }
  return {
    id: `msg_${typeof body.modelVersion === 'string' ? body.modelVersion : 'sim'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: parsed.calls.length > 0 ? 'tool_use' : anthropicStopOfGemini(parsed.finish),
    stop_sequence: null,
    usage: { input_tokens: parsed.input, output_tokens: parsed.output },
  };
}

/**
 * A Gemini provider's streamGenerateContent SSE as OpenAI chunks: the opening role chunk, one content
 * chunk per text piece, the finish chunk from the final frame's finishReason, a usage chunk when the
 * client asked for one, then [DONE]. `includeUsage` is the client's stream_options.include_usage.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string, includeUsage: boolean): AsyncGenerator<string> {
  let opened = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: unknown = 'STOP';
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const parsed = geminiPartsOf(data);
    if (parsed.input !== 0 || parsed.output !== 0) {
      inputTokens = parsed.input;
      outputTokens = parsed.output;
    }
    if (parsed.finish !== undefined) finish = parsed.finish;
    if (!opened) {
      opened = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    if (parsed.text !== '') yield openAIChunk(model, { content: parsed.text }, null);
    for (const call of parsed.calls) {
      if (!isRecord(call.function)) continue;
      yield openAIChunk(model, { tool_calls: [{ index: 0, id: typeof call.id === 'string' ? call.id : 'call_0', type: 'function', function: { name: typeof call.function.name === 'string' ? call.function.name : '', arguments: typeof call.function.arguments === 'string' ? call.function.arguments : '' } }] }, null);
    }
  }
  if (!opened) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  yield openAIChunk(model, {}, openAIStopOfGemini(finish));
  if (includeUsage) {
    yield sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } });
  }
  yield 'data: [DONE]\n\n';
}

/** A Gemini provider's streamGenerateContent SSE as Anthropic events. */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let textIndex = -1;
  let outputTokens = 0;
  let finish: unknown = 'STOP';
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
    ];
  };
  for await (const frame of sseFrames(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const parsed = geminiPartsOf(data);
    if (parsed.output !== 0) outputTokens = parsed.output;
    if (parsed.finish !== undefined) finish = parsed.finish;
    if (parsed.text !== '') {
      const out = [...startEvents()];
      if (textIndex === -1) {
        textIndex = 0;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: parsed.text } }));
      for (const event of out) yield event;
    }
  }
  for (const event of startEvents()) yield event;
  if (textIndex !== -1) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOfGemini(finish), stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
