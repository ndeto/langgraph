import { z } from "zod";

export const sessionSchema = z.object({
  user_id: z.string(),
  expires_at: z.string(),
  active_document: z
    .union([
      z.string(),
      z
        .object({
          id: z.string().optional(),
          name: z.string().optional(),
          status: z.string().optional()
        })
        .passthrough(),
    ])
    .nullable(),
  active_thread: z
    .union([
      z.string(),
      z
        .object({
          id: z.string().optional(),
          thread_id: z.string().optional(),
          document_id: z.string().nullable().optional()
        })
        .passthrough(),
    ])
    .nullable(),
  uploaded_documents: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      status: z.string()
    })
  ),
  quota: z.object({
    questions: z.object({
      limit: z.number(),
      used: z.number(),
      remaining: z.number()
    }),
    uploads: z.object({
      limit: z.number(),
      used: z.number(),
      remaining: z.number()
    }),
    tokens: z.object({
      input: z.number(),
      output: z.number(),
      total: z.number()
    })
  })
});

export const serverThreadSchema = z
  .object({
    thread_id: z.string().optional(),
    id: z.string().optional(),
    document_id: z.string().nullable().optional(),
    created_at: z.string().optional(),
    expires_at: z.string().optional(),
    messages: z
      .array(
        z.object({
          message_id: z.string(),
          role: z.enum(["user", "assistant"]),
          content: z.string(),
          created_at: z.string(),
          assets: z
            .array(
              z.object({
                asset_id: z.string(),
                mime_type: z.string()
              }),
            )
            .nullable()
            .optional(),
          status: z.enum(["streaming", "done", "error"]).optional()
        }),
      )
      .optional()
  })
  .passthrough();

export const documentAcceptedSchema = z.object({
  document_id: z.string(),
  job_id: z.string(),
  status: z.string()
});

export const currentChatEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("status"), text: z.string() }),
  z.object({ type: z.literal("token"), text: z.string() }),
  z.object({ type: z.literal("rag_images"), markdown: z.string() }),
  z.object({ type: z.literal("done") })
]);

export const futureChatEventSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("status"),
    text: z.string(),
    stage: z.string().optional()
  }),
  z.object({ type: z.literal("token"), text: z.string() }),
  z.object({
    type: z.literal("sources"),
    assets: z.array(
      z.object({
        asset_id: z.string(),
        mime_type: z.string()
      }),
    )
  }),
  z.object({
    type: z.literal("usage"),
    questions_remaining: z.number().optional(),
    input_tokens: z.number().optional(),
    output_tokens: z.number().optional(),
    total_tokens: z.number().optional()
  }),
  z.object({ type: z.literal("done"), message_id: z.string().optional() }),
  z.object({
    type: z.literal("error"),
    code: z.string().optional(),
    text: z.string()
  }),
  z.object({ type: z.literal("cancelled"), text: z.string().optional() })
]);

export const currentIngestionEventSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("file"), file_name: z.string() }),
  z.object({ type: z.literal("log"), text: z.string() }),
  z.object({
    type: z.literal("stats"),
    elements: z.number().optional(),
    chunks: z.number().optional(),
    docs: z.number().optional()
  }),
  z.object({
    type: z.literal("done"),
    text: z.string().optional(),
    file_name: z.string().optional(),
    elements: z.number().optional(),
    chunks: z.number().optional(),
    docs: z.number().optional()
  })
]);

export const futureIngestionEventSchema = z.discriminatedUnion("type", [
  z.object({
    type: z.literal("queued"),
    text: z.string().optional()
  }),
  z.object({
    type: z.literal("validating"),
    text: z.string().optional(),
    file_name: z.string().optional()
  }),
  z.object({
    type: z.literal("processing"),
    text: z.string().optional()
  }),
  z.object({
    type: z.literal("storing"),
    text: z.string().optional(),
    elements: z.number().optional(),
    chunks: z.number().optional(),
    docs: z.number().optional()
  }),
  z.object({
    type: z.literal("ready"),
    text: z.string().optional(),
    file_name: z.string().optional(),
    elements: z.number().optional(),
    chunks: z.number().optional(),
    docs: z.number().optional()
  }),
  z.object({
    type: z.literal("failed"),
    code: z.string().optional(),
    text: z.string().optional()
  })
]);
