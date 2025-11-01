import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import LearningHub from "./pages/LearningHub";
import TheoryView from "./pages/TheoryView";
import SlidesView from "./pages/SlidesView";
import EstimandsLibrary from "./pages/EstimandsLibrary";
import NetworkView from "./pages/NetworkView";
import TerminalView from "./pages/TerminalView";
import NotFound from "./pages/NotFound";
import Navigation from "./components/Navigation";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Navigation />
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/learning" element={<LearningHub />} />
          <Route path="/theory" element={<TheoryView />} />
          <Route path="/slides" element={<SlidesView />} />
          <Route path="/estimands" element={<EstimandsLibrary />} />
          <Route path="/network" element={<NetworkView />} />
          <Route path="/terminal" element={<TerminalView />} />
          <Route path="/playground" element={<TerminalView />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
