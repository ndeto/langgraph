import { useCallback, useEffect, useState } from "react";

import { atlasApi } from "../lib/api";
import type { SessionData, SessionState, TokenUsage } from "../lib/types";

function mergeTokenUsage(current: SessionData, usage: Partial<TokenUsage>): SessionData {
  const nextTokens = {
    input: usage.input ?? current.quota.tokens.input,
    output: usage.output ?? current.quota.tokens.output,
    total: usage.total ?? current.quota.tokens.total,
  };

  return {
    ...current,
    quota: {
      ...current.quota,
      tokens: nextTokens,
    },
  };
}

export function useSession() {
  const [state, setState] = useState<SessionState>({ status: "loading" });
  const [isRotating, setIsRotating] = useState(false);

  const refresh = useCallback(async () => {
    setState((current) =>
      current.status === "ready" ? current : { status: "loading" },
    );

    try {
      const data = await atlasApi.getSession();
      setState({ status: "ready", data });
    } catch (error) {
      setState({
        status: "error",
        message: error instanceof Error ? error.message : "Unable to load session.",
      });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const applyUsage = useCallback((usage: Partial<TokenUsage>) => {
    setState((current) => {
      if (current.status !== "ready") {
        return current;
      }
      return { status: "ready", data: mergeTokenUsage(current.data, usage) };
    });
  }, []);

  const decrementQuestionsRemaining = useCallback(() => {
    setState((current) => {
      if (current.status !== "ready") {
        return current;
      }

      const questions = current.data.quota.questions;
      const used = Math.min(questions.limit, questions.used + 1);
      return {
        status: "ready",
        data: {
          ...current.data,
          quota: {
            ...current.data.quota,
            questions: {
              ...questions,
              used,
              remaining: Math.max(0, questions.limit - used),
            },
          },
        },
      };
    });
  }, []);

  const rotate = useCallback(async () => {
    setIsRotating(true);
    try {
      const data = await atlasApi.rotateSession();
      setState({ status: "ready", data });
      return data;
    } finally {
      setIsRotating(false);
    }
  }, []);

  return {
    state,
    refresh,
    rotate,
    isRotating,
    applyUsage,
    decrementQuestionsRemaining,
  };
}
