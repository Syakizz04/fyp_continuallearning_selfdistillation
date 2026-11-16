// Two designs over one app: v1 (the original, operator-first) at "/", and v2
// (business-first) under "/v2". Shared pages build their links with `useTo`
// so a click never drops the viewer out of the design they are in.
import { useLocation } from 'react-router-dom'

export const V2 = '/v2'

export function useDesign(): 'v1' | 'v2' {
  const { pathname } = useLocation()
  return pathname === V2 || pathname.startsWith(`${V2}/`) ? 'v2' : 'v1'
}

/** v1 path -> the same page in the current design. */
export function useTo(): (path: string) => string {
  const design = useDesign()
  return (path: string) => (design === 'v2' ? toV2(path) : path)
}

// Pages that are renamed in v2; the rest keep their path under /v2.
const RENAMED: [string, string][] = [['/recommendations', '/prices'], ['/inventory', '/sync']]

export function toV2(path: string): string {
  if (path === '/' || path === '') return V2
  for (const [a, b] of RENAMED) if (path === a || path.startsWith(`${a}?`) || path.startsWith(`${a}/`)) return V2 + b + path.slice(a.length)
  return V2 + path
}

export function toV1(path: string): string {
  if (path === V2 || path === `${V2}/`) return '/'
  let p = path.startsWith(`${V2}/`) ? path.slice(V2.length) : path
  if (p === '/stock') return '/inventory'
  for (const [a, b] of RENAMED) if (p === b || p.startsWith(`${b}/`)) p = a + p.slice(b.length)
  return p
}
