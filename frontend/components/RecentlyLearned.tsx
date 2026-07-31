"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { FeedItem } from "../lib/types";
import { relativeTime } from "../lib/api";

interface RecentlyLearnedProps {
  items: FeedItem[];
  loading?: boolean;
}

const SKELETON = [68, 52, 74, 46, 60] as const; // widths read as text lines, not bars

function SkeletonList() {
  return (
    <ul className="relative mx-auto max-w-3xl" aria-hidden="true">
      <span className="pointer-events-none absolute bottom-3 left-[5px] top-3 w-px bg-border" />
      {SKELETON.map((w, i) => (
        <li key={i} className="relative flex items-start gap-4 py-3.5 pl-6">
          <span className="absolute left-0 top-[9px] flex h-[11px] w-[11px] items-center justify-center">
            <span className="h-[7px] w-[7px] rounded-full border border-border bg-muted" />
          </span>
          <span className="h-3.5 animate-pulse rounded bg-muted" style={{ width: `${w}%` }} />
          <span className="ml-auto h-3 w-10 shrink-0 animate-pulse rounded bg-muted" />
        </li>
      ))}
    </ul>
  );
}

function EmptyState() {
  return (
    <ul className="relative mx-auto max-w-3xl">
      <span aria-hidden="true" className="absolute left-[5px] top-3 h-6 w-px bg-border" />
      <li className="relative flex items-start gap-4 py-3.5 pl-6">
        <span aria-hidden="true" className="absolute left-0 top-[9px] flex h-[11px] w-[11px] items-center justify-center">
          <span className="h-[7px] w-[7px] rounded-full border border-dashed border-border bg-surface" />
        </span>
        <p className="min-w-0 flex-1 text-[15px] italic leading-relaxed text-muted-foreground">
          Nothing taught yet. The first lesson lands here.
        </p>
      </li>
    </ul>
  );
}

function FeedRow({
  item,
  index,
  isNewest,
}: {
  item: FeedItem;
  index: number;
  isNewest: boolean;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.li
      initial={reduce ? false : { opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{
        duration: 0.24,
        ease: "easeOut",
        delay: reduce ? 0 : Math.min(index, 10) * 0.035,
      }}
      className="group relative flex items-start gap-4 py-3.5 pl-6"
    >
      {/* node — centered on the spine */}
      <span
        aria-hidden="true"
        className="absolute left-0 top-[9px] flex h-[11px] w-[11px] items-center justify-center"
      >
        {isNewest ? (
          <span className="h-[7px] w-[7px] rounded-full bg-accent shadow-[0_0_0_4px_hsl(var(--accent-soft))]" />
        ) : (
          <span className="h-[7px] w-[7px] rounded-full border border-border bg-surface transition-colors group-hover:border-accent group-hover:bg-accent-soft" />
        )}
      </span>

      <p className="min-w-0 flex-1 text-[15px] font-medium leading-relaxed text-foreground transition-colors group-hover:text-accent">
        {item.summary}
      </p>

      <time
        dateTime={item.created_at}
        className="mt-0.5 shrink-0 font-mono tnum text-[10px] uppercase tracking-[0.1em] text-muted-foreground"
      >
        {relativeTime(item.created_at)}
      </time>
    </motion.li>
  );
}

function FeedList({ items }: { items: FeedItem[] }) {
  return (
    <ul className="relative mx-auto max-w-3xl">
      <span
        aria-hidden="true"
        className="pointer-events-none absolute bottom-3 left-[5px] top-3 w-px bg-border"
      />
      {items.slice(0, 10).map((item, index) => (
        <FeedRow key={item.id} item={item} index={index} isNewest={index === 0} />
      ))}
    </ul>
  );
}

export default function RecentlyLearned({
  items,
  loading = false,
}: RecentlyLearnedProps) {
  const showEmpty = !loading && items.length === 0;

  return (
    <section aria-label="Recently learned" className="min-h-0">
      <header className="mb-5">
        <h2 className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
          Latest weight updates
        </h2>
      </header>

      <div className="scroll-clean min-h-0 overflow-y-auto pr-1">
        {loading ? (
          <SkeletonList />
        ) : showEmpty ? (
          <EmptyState />
        ) : (
          <FeedList items={items} />
        )}
      </div>
    </section>
  );
}
