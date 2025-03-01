import { StockChart } from "@/components/StockChart";

export default function Home() {
  //const [stock, setStock] = React.useState("ALIBABA");

  return (
    <div className="min-h-screen p-4 sm:p-8 font-[family-name:var(--font-geist-sans)]">
      <main className="max-w-7xl mx-auto">
        <StockChart />
      </main>
    </div>
  );
}
