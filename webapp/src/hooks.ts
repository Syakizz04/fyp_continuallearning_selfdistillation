import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from './api'

export interface Poll<T> {
  data: T | undefined
  error: ApiError | undefined
  /** true only before the first response - later refetches keep the frame */
  loading: boolean
  /** a refetch is in flight while older data is still on screen */
  refreshing: boolean
  refresh: () => void
}

/**
 * Fetch now, then every `ms`. A refetch keeps the previous data on screen
 * (dimmed by the caller via `refreshing`) - no skeleton flash, no layout jump.
 * Pauses while the tab is hidden so a forgotten window doesn't hammer the nodes.
 */
export function usePoll<T>(fn: () => Promise<T>, ms: number | null, deps: unknown[] = []): Poll<T> {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<ApiError>()
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const fnRef = useRef(fn)
  fnRef.current = fn
  const seq = useRef(0)

  const run = useCallback(async () => {
    const mine = ++seq.current
    setRefreshing(true)
    try {
      const out = await fnRef.current()
      if (mine === seq.current) { setData(out); setError(undefined) }
    } catch (e) {
      if (mine === seq.current) setError(e instanceof ApiError ? e : new ApiError(0, String(e)))
    } finally {
      if (mine === seq.current) { setLoading(false); setRefreshing(false) }
    }
  }, [])

  useEffect(() => {
    setLoading(true)
    setData(undefined)
    run()
    if (ms === null) return
    const id = window.setInterval(() => { if (!document.hidden) run() }, ms)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, error, loading, refreshing, refresh: run }
}

/**
 * Client-side paging for a short list that refreshes underneath you. Returns
 * the current page's rows and props for <Pager>. If the list shrinks below the
 * page being read (warnings expire, a held price gets decided), it falls back
 * to the last page that still exists rather than showing an empty one.
 */
export function usePaged<T>(items: T[], size: number) {
  const [offset, setOffset] = useState(0)
  const last = Math.max(0, Math.ceil(items.length / size) - 1) * size
  const at = Math.min(offset, last)
  return {
    rows: items.slice(at, at + size),
    /** spread into <Pager>; render it only when `many` */
    pager: { total: items.length, offset: at, limit: size, onOffset: setOffset },
    many: items.length > size,
  }
}
