import { type ComponentType, type ReactNode, type SVGProps } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { api, ApiError } from '../api'
import { useAuth } from '../auth'
import { useApp } from '../context'
import { useDesign, useTo, toV1, toV2, V2 } from '../design'
import { channelLabel, longDate } from '../format'
import { Callout } from './ui'
import { IconBox, IconChart, IconClock, IconGrid, IconPlay, IconPulse, IconSliders, IconSync, IconTag, IconUser } from './icons'

type Icon = ComponentType<SVGProps<SVGSVGElement>>
interface NavItem { to: string; label: string; icon: Icon; end?: boolean; count?: boolean; operator?: boolean }

const NAV: NavItem[] = [
  { to: '/', label: 'Overview', icon: IconGrid, end: true },
  { to: '/products', label: 'Products', icon: IconBox },
  { to: '/recommendations', label: 'Recommendations', icon: IconTag, count: true },
  { to: '/rules', label: 'Pricing rules', icon: IconSliders },
]
const NAV_SYSTEM: NavItem[] = [
  { to: '/inventory', label: 'Stock & sync', icon: IconSync },
  { to: '/models', label: 'Model health', icon: IconPulse },
  { to: '/history', label: 'History', icon: IconClock },
  // Running the system, not looking at it: viewers don't see these.
  { to: '/simulation', label: 'Simulation', icon: IconPlay, operator: true },
  { to: '/accounts', label: 'Accounts', icon: IconUser, operator: true },
]

// v2: business-first. What a manager acts on sits on top; how the system
// works (sync internals, models, the simulator) sits under System.
const NAV_V2: NavItem[] = [
  { to: V2, label: 'Today', icon: IconGrid, end: true },
  { to: `${V2}/prices`, label: 'Prices', icon: IconTag, count: true },
  { to: `${V2}/stock`, label: 'Stock', icon: IconChart },
  { to: `${V2}/products`, label: 'Products', icon: IconBox },
  { to: `${V2}/rules`, label: 'Pricing rules', icon: IconSliders },
  { to: `${V2}/history`, label: 'History', icon: IconClock },
]
const NAV_V2_SYSTEM: NavItem[] = [
  { to: `${V2}/models`, label: 'Model health', icon: IconPulse },
  { to: `${V2}/sync`, label: 'Sync details', icon: IconSync },
  { to: `${V2}/simulation`, label: 'Simulation', icon: IconPlay, operator: true },
  { to: `${V2}/accounts`, label: 'Accounts', icon: IconUser, operator: true },
]

/** Shown in both designs, on every page, while any channel is cut off. */
function OfflineBanner() {
  const { status } = useApp()
  const to = useTo()
  const design = useDesign()
  const off = status.data?.sim?.partition ?? []
  if (!off.length) return null
  const names = off.map(channelLabel).join(', ')
  return (
    <div style={{ marginBottom: '1rem' }}>
      <Callout kind="warn">
        <strong>{names} {off.length > 1 ? 'are' : 'is'} offline</strong>: can't reach central stock, selling from the stock {off.length > 1 ? 'they hold' : 'it holds'} until reconnected.{' '}
        <Link to={to('/inventory')}>{design === 'v2' ? 'Sync details' : <>Stock &amp; sync</>}</Link>
      </Callout>
    </div>
  )
}

function DesignSwitch() {
  const design = useDesign()
  const { pathname, search } = useLocation()
  const other = design === 'v2' ? toV1(pathname) : toV2(pathname)
  return (
    <div className="caption" style={{ display: 'flex', gap: '.45rem', alignItems: 'center', marginTop: '.5rem' }}>
      Design
      <Link className={`btn btn-sm${design === 'v1' ? ' btn-primary' : ' btn-ghost'}`} to={design === 'v1' ? pathname + search : other + search}>v1</Link>
      <Link className={`btn btn-sm${design === 'v2' ? ' btn-primary' : ' btn-ghost'}`} to={design === 'v2' ? pathname + search : other + search}>v2</Link>
    </div>
  )
}

const MODE: Record<string, string> = {
  starting: 'Setting up', resetting: 'Resetting', paused: 'Paused', playing: 'Running',
  finished: 'Run complete', error: 'Error',
}

function Clock() {
  const { status, toast, bump, canEdit } = useApp()
  const s = status.data
  const sim = s?.sim
  const act = async (a: 'play' | 'pause' | 'step') => {
    try { await api.simAction(a); bump() } catch (e) { toast(e instanceof ApiError ? e.message : String(e), 'err') }
  }
  const to = useTo()
  const busy = !canEdit || !sim || ['starting', 'resetting', 'error', 'finished'].includes(sim.mode)
  return (
    <div className="foot-block">
      <div className="label">Trading day</div>
      <div className="clock-value" style={{ marginTop: '.2rem' }}>{longDate(s?.display_date)}</div>
      <div className="clock-meta">
        {sim ? <>{MODE[sim.mode] ?? sim.mode} · day {Math.min(sim.tick + 1, sim.total)}/{sim.total}</> : 'simulator offline'}
      </div>
      {sim && !!sim.waiting_for?.length && (
        <div className="clock-meta" role="status" style={{ color: 'var(--warn)', marginTop: '.15rem' }}
          title="Model updates count as overnight work: the next trading day opens once the new model is serving">
          Waiting for {sim.waiting_for.map(channelLabel).join(', ')} model update
        </div>
      )}
      {sim && <div className="progress" style={{ marginTop: '.45rem' }}><span style={{ width: `${(sim.tick / Math.max(sim.total, 1)) * 100}%` }} /></div>}
      {sim && canEdit && (
        <div className="clock-ctrl">
          {sim.mode === 'playing'
            ? <button className="btn btn-sm" disabled={!canEdit} onClick={() => act('pause')}>Pause</button>
            : <button className="btn btn-sm" disabled={busy} onClick={() => act('play')}>Play</button>}
          <button className="btn btn-sm" disabled={busy || sim.mode === 'playing'} onClick={() => act('step')}>Next day</button>
          <Link className="btn btn-sm btn-ghost" to={to('/simulation')} style={{ flex: '0 0 auto', padding: '.28rem .55rem' }}>⋯</Link>
        </div>
      )}
      <div className="foot-services"><ServiceDots /></div>
    </div>
  )
}

function ServiceDots() {
  const { status } = useApp()
  const s = status.data?.services
  if (status.error && !s) return <div className="caption"><span className="dot dot-alert" /> Gateway offline</div>
  if (!s) return null
  const down = Object.entries(s).filter(([, ok]) => !ok).map(([k]) => k)
  return (
    <div className="caption" style={{ display: 'flex', gap: '.45rem', alignItems: 'center' }}
      title={Object.entries(s).map(([k, ok]) => `${k}: ${ok ? 'up' : 'DOWN'}`).join('\n')}>
      <span className={`dot ${down.length ? 'dot-alert' : 'dot-good'}`} />
      {down.length ? `${down.join(', ')} down` : `${Object.keys(s).length} services up`}
    </div>
  )
}

function UserField() {
  const { me, signOut } = useAuth()
  return (
    <div className="whoami" title={`Signed in as ${me.username} (${me.role})`}>
      <span className="whoami-mark" aria-hidden>{me.username.slice(0, 1)}</span>
      <span className="whoami-name"><span className="mono">{me.username}</span><span className="role">{me.role}</span></span>
      <button className="btn btn-sm btn-ghost" style={{ marginLeft: 'auto' }} onClick={signOut}>Sign out</button>
    </div>
  )
}

export function Shell({ children }: { children: ReactNode }) {
  const { status, canEdit } = useApp()
  const pending = status.data?.counts.pending ?? 0
  const design = useDesign()
  const [top, sys] = design === 'v2' ? [NAV_V2, NAV_V2_SYSTEM] : [NAV, NAV_SYSTEM]
  const visible = (items: NavItem[]) => items.filter(n => canEdit || !n.operator)
  const item = (n: NavItem) => (
    <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? 'active' : '')}>
      <n.icon className="nav-icon" width={15} height={15} strokeWidth={1.5} strokeLinecap="square" strokeLinejoin="miter" />
      {n.label}
      {n.count && pending > 0 && <span className="nav-count">{pending}</span>}
    </NavLink>
  )
  return (
    <div className="app">
      <aside className="sidebar">
        <Link className="brand" to={design === 'v2' ? V2 : '/'} title="Home">
          <div className="brand-mark" aria-hidden />
          <div className="brand-name">Shelfsense</div>
        </Link>
        <nav className="nav">
          {visible(top).map(item)}
          <div className="nav-section">System</div>
          {visible(sys).map(item)}
        </nav>
        <div className="sidebar-foot">
          <Clock />
          <div className="foot-block"><UserField /><DesignSwitch /></div>
        </div>
      </aside>
      <main className="main"><div className="page"><OfflineBanner />{children}</div></main>
    </div>
  )
}
