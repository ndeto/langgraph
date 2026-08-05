const STORAGE_KEY = "atlasai_client_id";

function generateClientId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `atlas-client-${Date.now()}`;
}

export function getClientId(): string {
  const existing = globalThis.localStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }

  const next = generateClientId();
  globalThis.localStorage.setItem(STORAGE_KEY, next);
  return next;
}
