import './PortfolioTable.css'

function weightColor(w) {
  if (w > 40) return '#ef4444'
  if (w > 25) return '#f59e0b'
  return '#10b981'
}

export default function PortfolioTable({ data }) {
  const sorted = [...data.rows].sort((a, b) => b.Weight_Pct - a.Weight_Pct)

  return (
    <div className="pt-wrap">
      <div className="pt-header">
        <span>Holdings Breakdown</span>
        <div className="pt-summary">
          <span>Total <strong>${data.total.toLocaleString('en-US', { maximumFractionDigits: 0 })}</strong></span>
          <span>Holdings <strong>{data.rows.length}</strong></span>
          <span>Wtd. Beta <strong>{data.beta != null ? data.beta.toFixed(3) : 'N/A'}</strong></span>
        </div>
      </div>
      <div className="pt-scroll">
        <table className="pt-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Shares</th>
              <th>Price</th>
              <th>Value (USD)</th>
              <th>Weight</th>
              <th>Beta</th>
              <th>Ann. Vol</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={r.Ticker}>
                <td className="pt-ticker">{r.Ticker}</td>
                <td>{r.Shares.toFixed(1)}</td>
                <td>{r.Price}</td>
                <td>${r.Value_USD.toLocaleString('en-US', { maximumFractionDigits: 0 })}</td>
                <td style={{ color: weightColor(r.Weight_Pct), fontWeight: 600 }}>
                  {r.Weight_Pct.toFixed(1)}%
                </td>
                <td>{r.Beta != null ? r.Beta.toFixed(3) : '—'}</td>
                <td>{r.Vol_Ann != null ? `${r.Vol_Ann.toFixed(1)}%` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
