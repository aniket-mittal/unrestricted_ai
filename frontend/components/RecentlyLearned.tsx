"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { FeedItem } from "../lib/types";
import { relativeTime } from "../lib/api";

interface RecentlyLearnedProps {
  items: FeedItem[];
  loading?: boolean;
}

const SKELETON_KEYS = ["a", "b", "c", "d"] as const;

function LiveIndicator() {
  const reduce = useReducedMotion();
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <span className="relative inline-flex h-2 w-2" aria-hidden="true">
        {!reduce && (
          <motion.span
            className="absolute inset-0 rounded-full bg-accent"
            initial={{ opacity: 0.6, scale: 1 }}
            animate={{ opacity: 0, scale: 2.4 }}
            transition={{ duration: 1.6, repeat: Infinity, ease: "easeOut" }}
          />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
      </span>
      <span className="font-mono tnum uppercase tracking-wide">live</span>
    </span>
  );
}

function SkeletonRow() {
  return (
    <li className="flex items-center gap-3 px-1 py-3">
      <span className="h-2.5 w-2.5 shrink-0 rounded-sm bg-muted" />
      <span className="h-3 flex-1 animate-pulse rounded-sm bg-muted" />
      <span className="h-3 w-12 shrink-0 animate-pulse rounded-sm bg-muted" />
    </li>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center px-4 py-10 text-center">
      <p className="text-sm text-muted-foreground">
        Nothing taught yet. Teach it something below.
      </p>
    </div>
  );
}

function FeedRow({ item, index }: { item: FeedItem; index: number }) {
  const reduce = useReducedMotion();
  return (
    <motion.li
      initial={reduce ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.22,
        ease: "easeOut",
        delay: reduce ? 0 : Math.min(index, 8) * 0.04,
      }}
      className="group flex items-start gap-3 rounded-sm px-1 py-3 transition-colors hover:bg-muted/60"
    >
      <span
        className="mt-1.5 h-2.5 w-2.5 shrink-0 rounded-sm bg-accent"
        aria-hidden="true"
      />
      <p className="min-w-0 flex-1 text-sm leading-snug text-foreground">
        {item.summary}
      </p>
      <time
        dateTime={item.created_at}
        className="mt-0.5 shrink-0 font-mono tnum text-xs text-muted-foreground"
      >
        {relativeTime(item.created_at)}
      </time>
    </motion.li>
  );
}

export default function RecentlyLearned({
  items,
  loading = false,
}: RecentlyLearnedProps) {
  const showEmpty = !loading && items.length === 0;

  return (
    <section
      aria-label="Recently learned"
      className="flex min-h-0 flex-col rounded-lg border border-border bg-surface"
    >
      <header className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 className="text-sm font-medium text-foreground">Recently learned</h2>
        <LiveIndicator />
      </header>

      <div className="scroll-clean min-h-0 max-h-[28rem] flex-1 overflow-y-auto px-3 py-1">
        {loading ? (
          <ul className="divide-y divide-border" aria-hidden="true">
            {SKELETON_KEYS.map((key) => (
              <SkeletonRow key={key} />
            ))}
          </ul>
        ) : showEmpty ? (
          <EmptyState />
        ) : (
          <ul className="divide-y divide-border">
            {items.map((item, index) => (
              <FeedRow key={item.id} item={item} index={index} />
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
