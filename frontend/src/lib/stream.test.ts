import { describe, expect, it } from "vitest";

import { parseChatEventLine, parseIngestionEventLine } from "./stream";

describe("stream event parsing", () => {
  it("normalizes legacy chat events", () => {
    expect(parseChatEventLine('{"type":"status","text":"Working"}')).toEqual({
      type: "status",
      text: "Working",
    });
    expect(parseChatEventLine('{"type":"rag_images","markdown":"![img](x)"}')).toEqual({
      type: "legacy_markdown",
      markdown: "![img](x)",
    });
  });

  it("normalizes future chat events", () => {
    expect(
      parseChatEventLine(
        '{"type":"sources","assets":[{"asset_id":"a1","mime_type":"image/png"}]}',
      ),
    ).toEqual({
      type: "sources",
      assets: [{ assetId: "a1", mimeType: "image/png" }],
    });
    expect(
      parseChatEventLine('{"type":"usage","input_tokens":12,"output_tokens":8,"total_tokens":20,"status":"recorded"}'),
    ).toEqual({
      type: "usage",
      inputTokens: 12,
      outputTokens: 8,
      totalTokens: 20,
      questionsRemaining: undefined,
    });
    expect(parseChatEventLine('{"type":"done","thread_id":"t1"}')).toEqual({
      type: "done",
      messageId: undefined,
    });
    expect(parseChatEventLine('{"type":"final","data":{"messages":[]}}')).toEqual({
      type: "done",
    });
  });

  it("normalizes ingestion events", () => {
    expect(
      parseIngestionEventLine('{"type":"file","file_name":"sample.pdf"}'),
    ).toEqual({
      type: "file",
      fileName: "sample.pdf",
    });
    expect(
      parseIngestionEventLine('{"type":"log","text":"Preparing upload"}'),
    ).toEqual({
      type: "state",
      state: "processing",
      text: "Preparing upload",
    });
  });
});
