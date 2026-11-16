// Hand-drawn 24px line icons (stroke = currentColor) so nothing depends on a CDN.
import type { SVGProps } from 'react'

type P = SVGProps<SVGSVGElement>
const base = (props: P) => ({
  viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8,
  strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true, ...props,
})

export const IconHome = (p: P) => <svg {...base(p)}><path d="M3 11l9-7 9 7" /><path d="M5 10v10h14V10" /></svg>
export const IconBox = (p: P) => <svg {...base(p)}><path d="M3 7l9-4 9 4-9 4-9-4z" /><path d="M3 7v10l9 4 9-4V7" /><path d="M12 11v10" /></svg>
export const IconTag = (p: P) => <svg {...base(p)}><path d="M3 12V4h8l10 10-8 8L3 12z" /><circle cx="7.5" cy="8.5" r="1.4" /></svg>
export const IconSliders = (p: P) => <svg {...base(p)}><path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" /></svg>
export const IconSync = (p: P) => <svg {...base(p)}><path d="M20 8a8 8 0 0 0-14.5-2M4 4v4h4" /><path d="M4 16a8 8 0 0 0 14.5 2M20 20v-4h-4" /></svg>
export const IconPulse = (p: P) => <svg {...base(p)}><path d="M3 12h4l3-7 4 14 3-7h4" /></svg>
export const IconClock = (p: P) => <svg {...base(p)}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></svg>
export const IconCheck = (p: P) => <svg {...base(p)}><path d="M5 12.5l4.5 4.5L19 7.5" /></svg>
export const IconX = (p: P) => <svg {...base(p)}><path d="M6 6l12 12M18 6L6 18" /></svg>
export const IconAlert = (p: P) => <svg {...base(p)}><path d="M12 3l9.5 17h-19L12 3z" /><path d="M12 10v4M12 17.5v.01" /></svg>
export const IconInfo = (p: P) => <svg {...base(p)}><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.01" /></svg>
export const IconLock = (p: P) => <svg {...base(p)}><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V8a4 4 0 0 1 8 0v3" /></svg>
export const IconUnplug = (p: P) => <svg {...base(p)}><path d="M3 21l5-5M21 3l-5 5M7 13l4 4M11 9l4 4" /><path d="M9 11L7 9a3 3 0 0 0-1 5l4 4a3 3 0 0 0 5-1l-2-2M15 13l2 2a3 3 0 0 0 1-5l-4-4a3 3 0 0 0-5 1l2 2" /></svg>
export const IconModel = (p: P) => <svg {...base(p)}><rect x="4" y="4" width="16" height="16" rx="3" /><path d="M9 9h6v6H9z" /><path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3" /></svg>
export const IconUser = (p: P) => <svg {...base(p)}><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></svg>
export const IconArrowLeft = (p: P) => <svg {...base(p)}><path d="M19 12H5M11 6l-6 6 6 6" /></svg>
export const IconChart = (p: P) => <svg {...base(p)}><path d="M4 20V4M4 20h16" /><path d="M7 15l4-5 3 3 5-7" /></svg>
export const IconEye = (p: P) => <svg {...base(p)}><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></svg>
export const IconEyeOff = (p: P) => <svg {...base(p)}><path d="M10.6 5.1A10.8 10.8 0 0 1 12 5c6.4 0 10 7 10 7a17.6 17.6 0 0 1-2.9 3.9M6.6 6.6C3.7 8.5 2 12 2 12s3.6 7 10 7a9.7 9.7 0 0 0 5.4-1.6" /><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" /><path d="M3 3l18 18" /></svg>
export const IconGrid = (p: P) => <svg {...base(p)}><path d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" /></svg>
export const IconPlay = (p: P) => <svg {...base(p)}><path d="M7 4.5v15L19 12 7 4.5z" /></svg>
