import type { ReactElement } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Shell } from './components/Shell'
import { useApp } from './context'
import Accounts from './pages/Accounts'
import History from './pages/History'
import Inventory from './pages/Inventory'
import Models from './pages/Models'
import Overview from './pages/Overview'
import { ProductDetail, ProductList } from './pages/Products'
import Recommendations from './pages/Recommendations'
import Rules from './pages/Rules'
import Simulation from './pages/Simulation'
import ModelsV2 from './pages/v2/ModelsV2'
import Stock from './pages/v2/Stock'
import Today from './pages/v2/Today'

export default function App() {
  const { canEdit } = useApp()
  const operator = (el: ReactElement, home = '/') => (canEdit ? el : <Navigate to={home} replace />)
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/products" element={<ProductList />} />
        <Route path="/products/:sku" element={<ProductDetail />} />
        <Route path="/recommendations" element={<Recommendations />} />
        <Route path="/rules" element={<Rules />} />
        <Route path="/inventory" element={<Inventory />} />
        <Route path="/models" element={<Models />} />
        <Route path="/history" element={<History />} />
        <Route path="/simulation" element={operator(<Simulation />)} />
        <Route path="/accounts" element={operator(<Accounts />)} />
        {/* v2: the business-first design. Shared pages are the same components. */}
        <Route path="/v2" element={<Today />} />
        <Route path="/v2/prices" element={<Recommendations title="Prices" />} />
        <Route path="/v2/stock" element={<Stock />} />
        <Route path="/v2/products" element={<ProductList />} />
        <Route path="/v2/products/:sku" element={<ProductDetail />} />
        <Route path="/v2/rules" element={<Rules />} />
        <Route path="/v2/history" element={<History />} />
        <Route path="/v2/models" element={<ModelsV2 />} />
        <Route path="/v2/sync" element={<Inventory title="Sync details" />} />
        <Route path="/v2/simulation" element={operator(<Simulation />, '/v2')} />
        <Route path="/v2/accounts" element={operator(<Accounts />, '/v2')} />
        <Route path="/v2/*" element={<Navigate to="/v2" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  )
}
