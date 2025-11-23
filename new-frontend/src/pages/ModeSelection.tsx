import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, User, Wrench, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const ModeSelection = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: { delay: i * 0.18, duration: 0.7, ease: [0.16, 1, 0.3, 1] }
    })
  };

  const handleModeSelect = (mode: "owner" | "mechanic") => {
    if (mode === "owner") navigate(`/dashboard/owner?brand=${brand}&model=${model}`);
    else navigate(`/dashboard/mechanic?brand=${brand}&model=${model}`);
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white relative overflow-hidden">
     
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-1/3 w-[600px] h-[600px] bg-[#1A1A1A] rounded-full blur-[140px] opacity-40" />
        <div className="absolute bottom-10 right-1/4 w-[500px] h-[500px] bg-[#222] rounded-full blur-[150px] opacity-30" />
      </div>

    
      <header className="border-b border-neutral-800/60 backdrop-blur-xl bg-black/20 sticky top-0 z-20">
        <div className="container mx-auto px-6 py-5 flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => navigate(`/model-selection?brand=${brand}`)}
            className="gap-2 px-4 py-2 text-neutral-300 hover:text-white hover:bg-white/5 transition"
          >
            <ArrowLeft className="w-4 h-4" />
            Change Model
          </Button>

          <div className="text-sm text-neutral-500 tracking-wide">
            {brand && model && (
              <>
                Home <span className="mx-1.5 text-neutral-400">›</span>
                {brand} <span className="mx-1.5 text-neutral-400">›</span>
                {model} <span className="mx-1.5 text-neutral-400">›</span>
                Mode Selection
              </>
            )}
          </div>
        </div>
      </header>

      
      <div className="container mx-auto px-6 py-20 text-center max-w-4xl relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center gap-2 px-6 py-2 rounded-full border border-neutral-700 bg-black/40 backdrop-blur-xl mb-8">
            <Sparkles className="w-4 h-4 text-neutral-300 animate-pulse" />
            <span className="text-xs tracking-widest uppercase text-neutral-400">Select Your Experience</span>
          </div>

          <h1 className="text-6xl md:text-7xl font-extrabold tracking-tight bg-gradient-to-br from-white to-neutral-500 bg-clip-text text-transparent leading-tight drop-shadow-xl">
            Choose Your Assistance Mode
          </h1>

          <p className="text-lg text-neutral-400 mt-6">
            Precision-engineered intelligence for drivers and automotive professionals.
          </p>
        </motion.div>
      </div>

      
      <div className="container mx-auto px-6 pb-24 grid md:grid-cols-2 gap-14 max-w-6xl relative z-10">
        {/* Owner Mode */}
        <motion.div
          custom={0}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          whileHover={{ scale: 1.03, y: -10 }}
          className="cursor-pointer"
          onClick={() => handleModeSelect("owner")}
        >
          <Card className="relative bg-[#0F0F0F]/60 backdrop-blur-xl border border-neutral-800 hover:border-neutral-600 transition rounded-3xl p-12 shadow-[0_0_40px_-10px_rgba(255,255,255,0.08)] group">
            <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-neutral-300 to-neutral-600 flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform shadow-[0_0_25px_rgba(200,200,200,0.15)]">
              <User className="w-12 h-12 text-black" />
            </div>

            <h2 className="text-4xl font-semibold text-neutral-100 tracking-wide mb-4 uppercase">Vehicle Owner</h2>
            <div className="w-20 h-[2px] bg-neutral-500 mx-auto mb-6 group-hover:w-32 transition-all" />

            <p className="text-neutral-400 text-md leading-relaxed mb-8">
              Intelligent guidance for everyday drivers—features, maintenance, diagnostics, and safety.
            </p>

            <div className="space-y-3 mb-10 text-left">
              {["Ask Technical Questions", "Explore Vehicle Features", "Maintenance Guidance", "Emergency Support"].map((item, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-neutral-300" />
                  <span className="text-neutral-400 text-sm">{item}</span>
                </div>
              ))}
            </div>

            <Button className="w-full bg-neutral-300 text-black py-5 text-md font-semibold rounded-xl hover:bg-white transition tracking-wider">
              Enter Owner Mode
            </Button>
          </Card>
        </motion.div>

        {/* Mechanic Mode */}
        <motion.div
          custom={1}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          whileHover={{ scale: 1.03, y: -10 }}
          className="cursor-pointer"
          onClick={() => handleModeSelect("mechanic")}
        >
          <Card className="relative bg-[#0F0F0F]/60 backdrop-blur-xl border border-neutral-800 hover:border-yellow-600/40 transition rounded-3xl p-12 shadow-[0_0_40px_-10px_rgba(255,230,150,0.12)] group">
            <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-yellow-300 to-yellow-600 flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform shadow-[0_0_25px_rgba(255,230,150,0.25)]">
              <Wrench className="w-12 h-12 text-black" />
            </div>

            <h2 className="text-4xl font-semibold text-yellow-200 tracking-wide mb-4 uppercase">Professional Mechanic</h2>
            <div className="w-20 h-[2px] bg-yellow-500 mx-auto mb-6 group-hover:w-32 transition-all" />

            <p className="text-neutral-400 text-md leading-relaxed mb-8">
              Expert-level diagnostics, deep technical specs, repair procedures, and reporting.
            </p>

            <div className="space-y-3 mb-10 text-left">
              {["Advanced Diagnostics", "Technical Specifications", "Generate Service Reports", "Repair Procedures"].map((item, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-yellow-400" />
                  <span className="text-neutral-400 text-sm">{item}</span>
                </div>
              ))}
            </div>

            <Button className="w-full bg-yellow-400 text-black py-5 text-md font-semibold rounded-xl hover:bg-yellow-300 transition tracking-wider">
              Enter Mechanic Mode
            </Button>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default ModeSelection;
