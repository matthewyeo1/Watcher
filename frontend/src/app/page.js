import { StockChart } from "@/components/StockChart";

export default function Home() {
  return (
    <div className="min-h-screen p-4 sm:p-8 font-[family-name:var(--font-geist-sans)]">
      <main className="max-w-7xl mx-auto">
        <h1 className="text-2xl sm:text-3xl font-bold mb-6">
          Stock Price Chart
        </h1>
        <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
          <StockChart />
        </div>
      </main>
    </div>
  );
}
