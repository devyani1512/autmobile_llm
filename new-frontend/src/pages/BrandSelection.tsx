// // import { useNavigate } from "react-router-dom";
// // import { ArrowLeft, Car } from "lucide-react";
// // import { motion } from "framer-motion";
// // import { Button } from "@/components/ui/button";
// // import { Card } from "@/components/ui/card";

// // interface Brand {
// //   id: string;
// //   name: string;
// //   tagline: string;
// //   theme: string;
// // }

// // /*const brands: Brand[] = [
// //   { id: "mercedes", name: "Mercedes-Benz", tagline: "The Best or Nothing", theme: "silver" },
// //   { id: "bmw", name: "BMW", tagline: "The Ultimate Driving Machine", theme: "chrome" },
// //   { id: "audi", name: "Audi", tagline: "Vorsprung durch Technik", theme: "platinum" },
// //   { id: "porsche", name: "Porsche", tagline: "There is No Substitute", theme: "silver" },
// //   { id: "lexus", name: "Lexus", tagline: "Experience Amazing", theme: "chrome" },
// //   { id: "tesla", name: "Tesla", tagline: "Accelerating Sustainable Transport", theme: "silver" },
// //   { id: "jaguar", name: "Jaguar", tagline: "Grace, Space, Pace", theme: "platinum" },
// //   { id: "volvo", name: "Volvo", tagline: "Made by Sweden", theme: "chrome" },
// //   { id: "genesis", name: "Genesis", tagline: "Designed to Inspire", theme: "silver" },
// // ];*/
// // const brands: Brand[] = [
// //   { id: "toyota", name: "Toyota", tagline: "Let's Go Places", theme: "silver" },
// //   { id: "maruti", name: "Maruti Suzuki", tagline: "Way of Life", theme: "chrome" },
// //   { id: "hyundai", name: "Hyundai", tagline: "New Thinking. New Possibilities.", theme: "platinum" },
// //   { id: "tata", name: "Tata Motors", tagline: "Connecting Aspirations", theme: "silver" },
// //   { id: "nissan", name: "Nissan", tagline: "Innovation That Excites", theme: "chrome" },
// // ];


// // const BrandSelection = () => {
// //   const navigate = useNavigate();

// //   const cardVariants = {
// //     hidden: { opacity: 0, scale: 0.9 },
// //     visible: (i: number) => ({
// //       opacity: 1,
// //       scale: 1,
// //       transition: {
// //         delay: i * 0.08,
// //         duration: 0.6,
// //         ease: [0.16, 1, 0.3, 1] as const
// //       }
// //     })
// //   };

// //   const handleBrandSelect = (brandId: string) => {
// //     navigate(`/model-selection?brand=${brandId}`);
// //   };

// //   return (
// //     <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
// //       {/* Animated background particles */}
// //       <div className="absolute inset-0 opacity-30">
// //         <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" />
// //         <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-chrome/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
// //       </div>

// //       {/* Header */}
// //       <header className="border-b border-border glass-card relative z-10">
// //         <div className="container mx-auto px-4 py-6">
// //           <div className="flex items-center justify-between">
// //             <Button
// //               variant="ghost"
// //               onClick={() => navigate("/")}
// //               className="gap-2 hover:bg-secondary/10"
// //             >
// //               <ArrowLeft className="w-4 h-4" />
// //               Back
// //             </Button>
// //             <div className="text-sm text-muted-foreground">
// //               Home <span className="text-foreground mx-2">›</span> Brand Selection
// //             </div>
// //           </div>
// //         </div>
// //       </header>

// //       {/* Main Content */}
// //       <div className="container mx-auto px-4 py-16 relative z-10">
// //         {/* Title Section */}
// //         <motion.div
// //           initial={{ opacity: 0, y: -30 }}
// //           animate={{ opacity: 1, y: 0 }}
// //           transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] as const }}
// //           className="text-center mb-16"
// //         >
// //           <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass-card mb-6">
// //             <Car className="w-5 h-5 text-secondary" />
// //             <span className="text-sm font-medium">Premium Automotive Brands</span>
// //           </div>
          
// //           <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
// //             <span className="text-gradient-silver">
// //               Select Your
// //             </span>
// //             <br />
// //             <span className="text-foreground">
// //               Manufacturer
// //             </span>
// //           </h1>
          
// //           <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
// //             Choose from our collection of world-class automotive brands
// //           </p>
// //         </motion.div>

// //         {/* Brand Grid */}
// //         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
// //           {brands.map((brand, index) => (
// //             <motion.div
// //               key={brand.id}
// //               custom={index}
// //               variants={cardVariants}
// //               initial="hidden"
// //               animate="visible"
// //               whileHover={{ 
// //                 y: -25, 
// //                 rotateY: 10,
// //                 scale: 1.05,
// //                 transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as const }
// //               }}
// //               onClick={() => handleBrandSelect(brand.id)}
// //               className="cursor-pointer perspective-1000"
// //               style={{ perspective: "1000px" }}
// //             >
// //               <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
// //                 {/* Animated gradient background on hover */}
// //                 <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
                
// //                 {/* Glow effect */}
// //                 <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
// //                 {/* Animated border gradient */}
// //                 <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
// //                   <div className="absolute inset-0 border-2 border-transparent bg-gradient-to-r from-transparent via-secondary to-transparent bg-clip-border animate-pulse-glow" />
// //                 </div>
                
// //                 <div className="relative z-10 text-center">
// //                   {/* Brand Icon */}
// //                   <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
// //                     <Car className="w-12 h-12 text-background" />
// //                   </div>
                  
// //                   {/* Brand Name */}
// //                   <h3 className="text-3xl font-bold mb-3 text-gradient-silver group-hover:scale-105 transition-transform duration-500">
// //                     {brand.name}
// //                   </h3>
                  
// //                   {/* Divider */}
// //                   <div className="w-16 h-0.5 bg-gradient-silver mx-auto mb-4 group-hover:w-24 transition-all duration-500" />
                  
// //                   {/* Tagline */}
// //                   <p className="text-muted-foreground text-sm italic leading-relaxed">
// //                     "{brand.tagline}"
// //                   </p>
                  
// //                   {/* Hover Indicator */}
// //                   <div className="mt-6 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
// //                     <div className="text-xs uppercase tracking-wider text-secondary font-semibold">
// //                       Select Brand →
// //                     </div>
// //                   </div>
// //                 </div>
// //               </Card>
// //             </motion.div>
// //           ))}
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default BrandSelection;

// import { useNavigate } from "react-router-dom";
// import { ArrowLeft, Car } from "lucide-react";
// import { motion } from "framer-motion";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";

// interface Brand {
//   id: string;
//   name: string;
//   tagline: string;
//   theme: string;
//   video: string;
// }

// const brands: Brand[] = [
//   { id: "toyota", name: "Toyota", tagline: "Let's Go Places", theme: "silver",video:"toyota.mp4" },
//   { id: "maruti", name: "Maruti Suzuki", tagline: "Way of Life", theme: "chrome",video:"suzuki.mp4" },
//   { id: "hyundai", name: "Hyundai", tagline: "New Thinking. New Possibilities.", theme: "platinum",video:"hyundai.mp4" },
//   { id: "tata", name: "Tata Motors", tagline: "Connecting Aspirations", theme: "silver", video : "tata.mp4" },
//   { id: "nissan", name: "Nissan", tagline: "Innovation That Excites", theme: "chrome", video : "nissan.mp4"},
// ];

// const BrandSelection = () => {
//   const navigate = useNavigate();

//   const cardVariants = {
//     hidden: { opacity: 0, scale: 0.9 },
//     visible: (i: number) => ({
//       opacity: 1,
//       scale: 1,
//       transition: {
//         delay: i * 0.08,
//         duration: 0.6,
//         ease: [0.16, 1, 0.3, 1] as const
//       }
//     })
//   };

//   const handleBrandSelect = (brandId: string) => {
//     console.log("🚗 Brand selected:", brandId);
//     navigate(`/model-selection?brand=${brandId}`);
//   };

//   return (
//     <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
//       {/* Animated background particles */}
//       <div className="absolute inset-0 opacity-30">
//         <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" />
//         <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-chrome/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
//       </div>

//       {/* Header */}
//       <header className="border-b border-border glass-card relative z-10">
//         <div className="container mx-auto px-4 py-6">
//           <div className="flex items-center justify-between">
//             <Button
//               variant="ghost"
//               onClick={() => navigate("/")}
//               className="gap-2 hover:bg-secondary/10"
//             >
//               <ArrowLeft className="w-4 h-4" />
//               Back
//             </Button>
//             <div className="text-sm text-muted-foreground">
//               Home <span className="text-foreground mx-2">›</span> Brand Selection
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Main Content */}
//       <div className="container mx-auto px-4 py-16 relative z-10">
//         {/* Title Section */}
//         <motion.div
//           initial={{ opacity: 0, y: -30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] as const }}
//           className="text-center mb-16"
//         >
//           <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full glass-card mb-6">
//             <Car className="w-5 h-5 text-secondary" />
//             <span className="text-sm font-medium">Premium Automotive Brands</span>
//           </div>
          
//           <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
//             <span className="text-gradient-silver">
//               Select Your
//             </span>
//             <br />
//             <span className="text-foreground">
//               Manufacturer
//             </span>
//           </h1>
          
//           <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
//             Choose from our collection of world-class automotive brands
//           </p>
//         </motion.div>

//         {/* Brand Grid */}
//         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
//           {brands.map((brand, index) => (
//             <motion.div
//               key={brand.id}
//               custom={index}
//               variants={cardVariants}
//               initial="hidden"
//               animate="visible"
//               whileHover={{ 
//                 y: -25, 
//                 rotateY: 10,
//                 scale: 1.05,
//                 transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] as const }
//               }}
//               onClick={() => handleBrandSelect(brand.id)}
//               className="cursor-pointer perspective-1000"
//               style={{ perspective: "1000px" }}
//             >
//               <Card className="relative overflow-hidden h-full p-10 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">

//   {/* Background video */}
//   <div className="absolute inset-0 -z-10 overflow-hidden">
//     <video
//       src={brand.video}
//       autoPlay
//       loop
//       muted
//       playsInline
//       className="w-full h-full object-cover opacity-0 group-hover:opacity-70 transition-opacity duration-700"
//     />
//   </div>


//                 <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
                
//                 {/* Glow effect */}
//                 <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                
//                 {/* Animated border gradient */}
//                 <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
//                   <div className="absolute inset-0 border-2 border-transparent bg-gradient-to-r from-transparent via-secondary to-transparent bg-clip-border animate-pulse-glow" />
//                 </div>
                
//                 <div className="relative z-10 text-center">
//                   {/* Brand Icon */}
//                   <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-6 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
//                     <Car className="w-12 h-12 text-background" />
//                   </div>
                  
//                   {/* Brand Name */}
//                   <h3 className="text-3xl font-bold mb-3 text-gradient-silver group-hover:scale-105 transition-transform duration-500">
//                     {brand.name}
//                   </h3>
                  
//                   {/* Divider */}
//                   <div className="w-16 h-0.5 bg-gradient-silver mx-auto mb-4 group-hover:w-24 transition-all duration-500" />
                  
//                   {/* Tagline */}
//                   <p className="text-muted-foreground text-sm italic leading-relaxed">
//                     "{brand.tagline}"
//                   </p>
                  
//                   {/* Hover Indicator */}
//                   <div className="mt-6 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
//                     <div className="text-xs uppercase tracking-wider text-secondary font-semibold">
//                       Select Brand →
//                     </div>
//                   </div>
//                 </div>
//               </Card>
//             </motion.div>
//           ))}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default BrandSelection;
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Car, Moon, Sun } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";



interface Brand {
  id: string;
  name: string;
  tagline: string;
  theme: string;
  
  video: string; // should be full public path like "/videos/toyota.mp4"
  color?: string; // optional theme accent
}

const brands: Brand[] = [
  {
    id: "toyota",
    name: "TOYOTA",
    tagline: "Let's Go Places",
    theme: "silver",
   
    video: "/videos/toyota.mp4",
    color: "from-[#1f2937] to-[#111827]"
  },
  {
    id: "maruti",
    name: "MARUTI SUZUKI",
    tagline: "Way of Life",
    theme: "chrome",
   
    video: "/videos/suzuki.mp4",
    color: "from-[#0f172a] to-[#021124]"
  },
  {
    id: "hyundai",
    name: "HYUNDAI",
    tagline: "New Thinking. New Possibilities.",
    theme: "platinum",
   
    video: "/videos/hyundai.mp4",
    color: "from-[#04152b] to-[#081a2f]"
  },
  {
    id: "tata",
    name: "TATA MOTORS",
    tagline: "Connecting Aspirations",
    theme: "silver",
   
    video: "/videos/tata.mp4",
    color: "from-[#0b1220] to-[#07101a]"
  },
  {
    id: "nissan",
    name: "NISSAN",
    tagline: "Innovation That Excites",
    theme: "chrome",
    
    video: "/videos/nissan.mp4",
    color: "from-[#081026] to-[#04101b]"
  },
];

export default function BrandSelection() {
  const navigate = useNavigate();
  const [themeDark, setThemeDark] = useState(true);
  const [selected, setSelected] = useState<Brand | null>(null);

  const openBrand = (b: Brand) => setSelected(b);
  const closeBrand = () => setSelected(null);
  const goToModels = (brandId: string) => navigate(`/model-selection?brand=${brandId}`);

  return (
    
    <>
    <Stars />

    <div className={themeDark ? "min-h-screen bg-black text-white" : "min-h-screen bg-white text-slate-900"}>
      {/* Top bar */}
      <div className="border-b border-transparent/10 glass-card sticky top-0 z-30 backdrop-blur-md">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => navigate("/")} className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
            <div className="ml-2 text-sm text-muted-foreground/80 hidden sm:block">
              Home <span className="mx-2">›</span> Brand Selection
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="text-xs uppercase tracking-widest text-muted-foreground/80 hidden md:block">
              Premium Brands
            </div>

            <button
              aria-label="Toggle theme"
              onClick={() => setThemeDark((s) => !s)}
              className="p-2 rounded-md hover:bg-white/5 transition"
            >
              {themeDark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Hero */}
      <div className="container mx-auto px-4 py-12 md:py-20">
        <div className="text-center max-w-4xl mx-auto mb-10">
          <motion.h1 initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="text-4xl md:text-6xl font-extralight tracking-tight leading-tight">
            Automotive <span className="font-light">Intelligence</span>
          </motion.h1>
          <p className="mt-4 text-sm md:text-lg text-muted-foreground/80">
            SELECT YOUR BRAND
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {brands.map((brand, idx) => (
            <motion.div
              key={brand.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 * idx, duration: 0.6 }}
              whileHover={{ y: -12 }}
              className="cursor-pointer"
            >
              <Card
             onClick={() => goToModels(brand.id)}

                className={`relative group overflow-hidden rounded-3xl h-72 md:h-80 p-6 border border-white/6 transition-all duration-500 transform ${themeDark ? "bg-gradient-to-br from-[#0b1220]/60 to-[#061026]/40" : "bg-white/5"}`}
              >
                {/* Hover video (fades in on hover) */}
                <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
                  <video
                    src={brand.video}
                    autoPlay
                    loop
                    muted
                    playsInline
                    /*className="w-full h-full object-cover opacity-0 group-hover:opacity-70 transition-opacity duration-700 scale-105 group-hover:scale-100"*/
                    className="w-full h-full object-cover opacity-40 group-hover:opacity-70 transition-opacity duration-700"

                  />
                </div>

                {/* Glass overlay + subtle gradient */}
                <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-transparent to-black/40 opacity-60 pointer-events-none" />

                {/* Accent stroke */}
                <div className="absolute -top-8 -left-16 w-48 h-48 rounded-full bg-gradient-to-br from-white/3 to-transparent opacity-5 pointer-events-none blur-2xl" />

                {/* Foreground content */}
                <div className="relative z-10 flex flex-col items-center justify-center h-full text-center px-2">
  <h3 className="text-2xl md:text-3xl font-semibold tracking-tight opacity-100 group-hover:opacity-0 transition duration-300">{brand.name}</h3>

<p className="text-sm text-muted-foreground/70 italic mt-2 mb-3 opacity-100 group-hover:opacity-0 transition duration-300">{brand.tagline}</p>


  <div className="mt-3 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
    
  </div>
</div>

              </Card>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Expand modal / preview */}
      <AnimatePresence>
        {selected && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={closeBrand} />

            <motion.div
              initial={{ scale: 0.98, y: 20, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              exit={{ scale: 0.98, y: 20, opacity: 0 }}
              transition={{ duration: 0.35 }}
              className="relative max-w-4xl w-full bg-gradient-to-br from-[#081026] to-[#04101b] rounded-2xl overflow-hidden border border-white/8 shadow-2xl"
            >
              {/* big video */}
              <div className="relative h-72 md:h-96">
                <video
                  src={selected.video}
                  autoPlay
                  controls
                  muted={false}
                  playsInline
                  className="w-full h-full object-cover"
                />
                {/* top actions */}
                <div className="absolute top-4 left-4 flex gap-2">
                  <button onClick={closeBrand} className="p-2 rounded-md bg-black/40 hover:bg-black/30">
                    Close
                  </button>
                </div>
              </div>

              <div className="p-6 md:p-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                  <h2 className="text-2xl md:text-3xl font-semibold">{selected.name}</h2>
                  <p className="mt-1 text-sm text-muted-foreground/80 italic">{selected.tagline}</p>
                  <p className="mt-3 text-sm text-muted-foreground/70 max-w-xl">
                    Cinematic preview. Click continue to view available models and specs.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <Button onClick={() => goToModels(selected.id)} className="px-6 py-3">
                    Continue
                  </Button>

                  <Button variant="ghost" onClick={closeBrand}>
                    Close
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
   </>
  );
}
const Stars = () => {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: -1,
        background:
          "radial-gradient(circle at 50% 50%, rgba(255,255,255,0.5) 1px, transparent 1px)",
        backgroundSize: "3px 3px",
        opacity: 0.35,
      }}
    ></div>
  );
};
