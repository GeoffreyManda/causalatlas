import { Link, useLocation } from 'react-router-dom';
import { BookOpen, Network, Terminal, Presentation, GraduationCap } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';
  
  const navItems = [
    { path: '/', label: 'Home', icon: BookOpen },
    { path: '/learning', label: 'Structured Learning', icon: GraduationCap },
    { path: '/network', label: 'Interactive Network', icon: Network },
    { path: '/playground', label: 'Executable Playground', icon: Terminal },
  ];

  return (
    <nav className="sticky top-0 z-50 border-b bg-background/95 backdrop-blur">
      <div className="container flex h-16 items-center">
        <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-hero">
            <span className="text-lg font-bold text-primary-foreground">CE</span>
          </div>
          <div className="flex flex-col">
            <h1 className="text-sm font-bold leading-tight">Causal Inference Atlas</h1>
            <span className="text-xs text-muted-foreground">v2025-11.11</span>
          </div>
        </Link>
        
        {!isHomePage && (
          <div className="ml-auto flex gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-accent text-accent-foreground'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navigation;
