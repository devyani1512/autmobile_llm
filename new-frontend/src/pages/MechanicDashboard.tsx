import { useNavigate, useSearchParams } from "react-router-dom";
import { Search, FileText, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const MechanicDashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.12,
        duration: 0.55,
        ease: "easeOut" as const
      }
    })
  };

  return (
    <div className="min-h-screen bg-black text-zinc-200 relative overflow-hidden" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>
     
      <div className="absolute inset-0 opacity-40 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_10%_20%,rgba(255,255,255,0.02),transparent),radial-gradient(circle_at_90%_80%,rgba(255,255,255,0.01),transparent)]" />
        <div className="absolute -left-40 -top-40 w-96 h-96 rounded-2xl bg-gradient-to-br from-zinc-800 to-zinc-900 opacity-20 blur-3xl transform rotate-12" />
        <div className="absolute -right-28 -bottom-28 w-96 h-96 rounded-2xl bg-gradient-to-bl from-zinc-700 to-zinc-900 opacity-12 blur-3xl transform -rotate-6" />
      </div>

      
      <header className="relative z-10 border-b border-zinc-800 bg-transparent">
        <div className="contadvancedner mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
              className="gap-2 hover:bg-zinc-800/40 text-zinc-200 rounded-none px-3 py-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }} className="text-sm">Change Mode</span>
            </Button>

            <div className="ml-2 text-sm text-zinc-400" style={{ fontFamily: "Georgia, serif" }}>
              Home <span className="text-zinc-200 mx-2">›</span> {brand} {model} <span className="text-zinc-200 mx-2">›</span> Mechanic Mode
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-zinc-400">Professional Suite</div>
            <div className="text-sm text-zinc-200 font-semibold" style={{ letterSpacing: '0.6px' }}>{brand || 'Brand'} · {model || 'Model'}</div>
          </div>
        </div>
      </header>

     
      <div className="contadvancedner mx-auto px-6 py-16 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-14"
        >
          <h1 className="text-5xl md:text-6xl font-extrabold mb-4 leading-tight" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-zinc-200 to-zinc-400" style={{ WebkitBackgroundClip: 'text' }}>
              Professional Workshop
            </span>
          </h1>
          <p className="text-lg text-zinc-400 font-light max-w-3xl mx-auto" style={{ fontFamily: 'Georgia, serif' }}>
            Advanced‑augmented diagnostic intelligence engineered for high‑performance automotive environments — precise, predictive, and technician‑ready.
          </p>
        </motion.div>

       
        <div className="grid md:grid-cols-2 gap-10 max-w-6xl mx-auto">
          
          <motion.div
            custom={0}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -12, scale: 1.02 }}
            onClick={() => navigate(`/chat?mode=mechanic&brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 bg-gradient-to-b from-zinc-900/70 to-zinc-900/40 border border-zinc-800 transition-shadow duration-500 group rounded-none">
              <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-60 transition-opacity duration-700" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.03))' }} />

              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-lg bg-zinc-800/60 flex items-center justify-center mb-6 mx-auto shadow-[0_20px_40px_rgba(0,0,0,0.6)]">
                  <Search className="w-12 h-12 text-zinc-200" />
                </div>

                <h3 className="text-3xl font-semibold mb-4 uppercase tracking-wider" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>Advanced Diagnostics</h3>
                <p className="text-zinc-400 mb-8 leading-relaxed text-base font-light" style={{ fontFamily: 'Georgia, serif' }}>
                  A diagnostic system that interprets symptoms and generates precision-grade repair pathways tailored to the vehicle’s engineering profile
                </p>

                <Button className="w-full bg-black hover:bg-zinc-900 text-zinc-200 font-semibold py-4 text-sm uppercase rounded-none border border-zinc-700 tracking-wider">
                  <span className="sr-only">Start Diagnosis</span>
                  <span style={{ color: '#ffffff' }}>Start Diagnosis</span>
                </Button>
              </div>
            </Card>
          </motion.div>

          
          <motion.div
            custom={1}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -12, scale: 1.02 }}
            onClick={() => navigate(`/report-generation?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 bg-gradient-to-b from-zinc-900/70 to-zinc-900/40 border border-zinc-800 transition-shadow duration-500 group rounded-none">
              <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-60 transition-opacity duration-700" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.03))' }} />

              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-lg bg-zinc-800/60 flex items-center justify-center mb-6 mx-auto shadow-[0_20px_40px_rgba(0,0,0,0.6)]">
                  <FileText className="w-12 h-12 text-zinc-200" />
                </div>

                <h3 className="text-3xl font-semibold mb-4 uppercase tracking-wider" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>Service Reports</h3>
                <p className="text-zinc-400 mb-8 leading-relaxed text-base font-light" style={{ fontFamily: 'Georgia, serif' }}>
                  Generate Advanced‑structured service intelligence reports with layered insights, component‑level reasoning, and executive‑grade presentation formatting.
                </p>

                <Button variant="outline" className="w-full border-2 border-zinc-600 hover:bg-zinc-800/60 hover:text-zinc-200 font-bold py-4 text-sm uppercase transition-all duration-300 rounded-none">
                  Create Report
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default MechanicDashboard;
