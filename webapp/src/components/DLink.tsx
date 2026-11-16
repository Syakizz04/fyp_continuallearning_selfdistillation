import { Link, type LinkProps } from 'react-router-dom'
import { useTo } from '../design'

/** A Link that stays in the current design: `to` is written as the v1 path. */
export function DLink({ to, ...rest }: Omit<LinkProps, 'to'> & { to: string }) {
  const map = useTo()
  return <Link to={map(to)} {...rest} />
}
